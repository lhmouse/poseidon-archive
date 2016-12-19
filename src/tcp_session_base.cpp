// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2016, LH_Mouse. All wrongs reserved.

#include "precompiled.hpp"
#include "tcp_session_base.hpp"
#include "ssl_filter_base.hpp"
#include "ip_port.hpp"
#include <sys/types.h>
#include <sys/socket.h>
#include <fcntl.h>
#include <errno.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include "singletons/timer_daemon.hpp"
#include "log.hpp"
#include "atomic.hpp"
#include "endian.hpp"
#include "string.hpp"
#include "time.hpp"
#include "system_exception.hpp"
#include "epoll.hpp"
#include "job_base.hpp"
#include "profiler.hpp"
#include "checked_arithmetic.hpp"

namespace Poseidon {

TcpSessionBase::DelayedShutdownGuard::DelayedShutdownGuard(boost::weak_ptr<TcpSessionBase> weak)
	: m_weak(STD_MOVE(weak))
{
	const AUTO(session, m_weak.lock());
	if(!session){
		return;
	}
	atomic_add(session->m_delayed_shutdown_guard_count, 1, ATOMIC_RELAXED);
}
TcpSessionBase::DelayedShutdownGuard::~DelayedShutdownGuard(){
	const AUTO(session, m_weak.lock());
	if(!session){
		return;
	}
	if(atomic_sub(session->m_delayed_shutdown_guard_count, 1, ATOMIC_RELAXED) == 0){
		if(atomic_load(session->m_shutdown_write, ATOMIC_CONSUME)){
			atomic_store(session->m_really_shutdown_write, true, ATOMIC_RELEASE);
			session->notify_epoll_writeable();
		}
	}
}

void TcpSessionBase::shutdown_timer_proc(const boost::weak_ptr<TcpSessionBase> &weak, boost::uint64_t now){
	const AUTO(session, weak.lock());
	if(!session){
		return;
	}
	if(now < session->m_shutdown_time){
		return;
	}

	if(session->get_send_buffer_size() != 0){
		LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_INFO, "Send buffer is not empty. Retry later...");
		return;
	}

	LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_DEBUG, "Connection timed out: remote = ", session->get_remote_info_nothrow());
	atomic_store(session->m_timed_out, true, ATOMIC_RELEASE);
	session->force_shutdown();
}

TcpSessionBase::TcpSessionBase(UniqueFile socket)
	: m_socket(STD_MOVE(socket)), m_created_time(get_fast_mono_clock())
	, m_peer_info()
	, m_connected(false), m_connected_notified(false)
	, m_shutdown_read(false), m_read_hup_notified(false), m_shutdown_write(false), m_really_shutdown_write(false)
	, m_timed_out(false), m_throttled(false)
	, m_delayed_shutdown_guard_count(0)
	, m_shutdown_time(0)
{
	const int flags = ::fcntl(m_socket.get(), F_GETFL);
	if(flags == -1){
		const int err_code = errno;
		LOG_POSEIDON_ERROR("Could not get fcntl flags on socket.");
		DEBUG_THROW(SystemException, err_code);
	}
	if(::fcntl(m_socket.get(), F_SETFL, flags | O_NONBLOCK) != 0){
		const int err_code = errno;
		LOG_POSEIDON_ERROR("Could not set fcntl flags on socket.");
		DEBUG_THROW(SystemException, err_code);
	}
}
TcpSessionBase::~TcpSessionBase(){
	LOG_POSEIDON_INFO("Destroying TCP session: remote = ", get_remote_info_nothrow());
}

bool TcpSessionBase::has_connected() const NOEXCEPT {
	return atomic_load(m_connected, ATOMIC_CONSUME);
}
void TcpSessionBase::set_connected(){
	atomic_store(m_connected, true, ATOMIC_RELEASE);
}
bool TcpSessionBase::is_connected_notified() const NOEXCEPT {
	return atomic_load(m_connected_notified, ATOMIC_CONSUME);
}
void TcpSessionBase::notify_connected() NOEXCEPT {
	PROFILE_ME;

	const bool old = atomic_exchange(m_connected_notified, true, ATOMIC_ACQ_REL);
	if(!old){
		on_connect();
	}
}

void TcpSessionBase::init_ssl(Move<boost::scoped_ptr<SslFilterBase> > ssl_filter){
	swap(m_ssl_filter, ssl_filter);
}

bool TcpSessionBase::is_read_hup_notified() const NOEXCEPT {
	return atomic_load(m_read_hup_notified, ATOMIC_CONSUME);
}
void TcpSessionBase::notify_read_hup() NOEXCEPT {
	PROFILE_ME;

	const bool old = atomic_exchange(m_read_hup_notified, true, ATOMIC_ACQ_REL);
	if(!old){
		on_read_hup();
	}
}

void TcpSessionBase::set_epoll(boost::weak_ptr<Epoll> epoll) NOEXCEPT {
	const Mutex::UniqueLock lock(m_buffer_mutex);
	const AUTO(old_epoll, m_epoll.lock());
	if(old_epoll){
		old_epoll->notify_unlinked(this);
	}
	m_epoll = STD_MOVE(epoll);
}
void TcpSessionBase::notify_epoll_writeable() NOEXCEPT {
	const AUTO(epoll, m_epoll.lock());
	if(epoll){
		epoll->notify_writeable(this);
	}
}

void TcpSessionBase::fetch_peer_info() const {
	const Mutex::UniqueLock lock(m_peer_info.mutex);
	if(m_peer_info.remote.port == 0){
		m_peer_info.remote = get_remote_ip_port_from_fd(m_socket.get());
		LOG_POSEIDON_DEBUG("TCP session remote info = ", m_peer_info.remote);
	}
	if(m_peer_info.local.port == 0){
		m_peer_info.local = get_local_ip_port_from_fd(m_socket.get());
		LOG_POSEIDON_DEBUG("TCP session local info = ", m_peer_info.local);
	}
}

TcpSessionBase::SyncIoResult TcpSessionBase::sync_read_and_process(void *hint, unsigned long hint_size){
	PROFILE_ME;

	SyncIoResult ret;
	if(m_ssl_filter){
		ret.bytes_transferred = m_ssl_filter->read(hint, hint_size);
	} else {
		ret.bytes_transferred = ::recv(m_socket.get(), hint, hint_size, MSG_NOSIGNAL);
	}
	ret.err_code = errno;

	if(ret.bytes_transferred > 0){
		fetch_peer_info();

		const AUTO(bytes, static_cast<std::size_t>(ret.bytes_transferred));
		LOG_POSEIDON_TRACE("Read ", bytes, " byte(s) from ", get_remote_info());

		on_read_avail(StreamBuffer(hint, bytes));
	}

	return ret;
}
TcpSessionBase::SyncIoResult TcpSessionBase::sync_write(void *hint, unsigned long hint_size){
	PROFILE_ME;

	std::size_t bytes_avail;
	{
		const Mutex::UniqueLock lock(m_buffer_mutex);
		bytes_avail = m_send_buffer.peek(hint, hint_size);
	}

	SyncIoResult ret;
	if(m_ssl_filter){
		ret.bytes_transferred = m_ssl_filter->write(hint, bytes_avail);
	} else {
		ret.bytes_transferred = ::send(m_socket.get(), hint, bytes_avail, MSG_NOSIGNAL);
	}
	ret.err_code = errno;

	if(ret.bytes_transferred > 0){
		fetch_peer_info();

		const AUTO(bytes, static_cast<std::size_t>(ret.bytes_transferred));
		LOG_POSEIDON_TRACE("Wrote ", bytes, " byte(s) to ", get_remote_info());

		const Mutex::UniqueLock lock(m_buffer_mutex);
		m_send_buffer.discard(bytes);
		bytes_avail = m_send_buffer.size();
	}

	if((bytes_avail == 0) && atomic_load(m_really_shutdown_write, ATOMIC_CONSUME)){
		if(m_ssl_filter){
			m_ssl_filter->send_fin();
		} else {
			::shutdown(m_socket.get(), SHUT_WR);
		}
	}
	return ret;
}
std::size_t TcpSessionBase::get_send_buffer_size(Mutex::UniqueLock *lock) const {
	Mutex::UniqueLock new_lock(m_buffer_mutex);
	const AUTO(size, m_send_buffer.size());
	if(lock){
		lock->swap(new_lock);
	}
	return size;
}

void TcpSessionBase::on_connect(){
}

void TcpSessionBase::on_read_hup() NOEXCEPT {
}
void TcpSessionBase::on_close(int err_code) NOEXCEPT {
	(void)err_code;

	m_connected = false;
}

bool TcpSessionBase::send(StreamBuffer buffer){
	PROFILE_ME;

	if(atomic_load(m_really_shutdown_write, ATOMIC_CONSUME)){
		LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_DEBUG,
			"Connection has been shut down for writing: remote = ", get_remote_info_nothrow());
		return false;
	}

	const Mutex::UniqueLock lock(m_buffer_mutex);
	if(!buffer.empty()){
		m_send_buffer.splice(buffer);
	}
	notify_epoll_writeable();
	return true;
}

bool TcpSessionBase::has_been_shutdown_read() const NOEXCEPT {
	return atomic_load(m_shutdown_read, ATOMIC_CONSUME);
}
bool TcpSessionBase::shutdown_read() NOEXCEPT {
	PROFILE_ME;

	bool ret = !atomic_load(m_shutdown_read, ATOMIC_CONSUME);
	if(ret){
		ret = !atomic_exchange(m_shutdown_read, true, ATOMIC_ACQ_REL);
		::shutdown(m_socket.get(), SHUT_RD);
	}
	return ret;
}
bool TcpSessionBase::has_been_shutdown_write() const NOEXCEPT {
	return atomic_load(m_shutdown_write, ATOMIC_CONSUME);
}
bool TcpSessionBase::shutdown_write() NOEXCEPT {
	PROFILE_ME;

	bool ret = !atomic_load(m_shutdown_write, ATOMIC_CONSUME);
	if(ret){
		ret = !atomic_exchange(m_shutdown_write, true, ATOMIC_ACQ_REL);
		const DelayedShutdownGuard guard(virtual_shared_from_this<TcpSessionBase>());
	}
	return ret;
}
void TcpSessionBase::force_shutdown() NOEXCEPT {
	PROFILE_ME;

	atomic_store(m_shutdown_read, true, ATOMIC_RELEASE);
	atomic_store(m_shutdown_write, true, ATOMIC_RELEASE);

	::linger lng;
	lng.l_onoff = 1;
	lng.l_linger = 0;
	::setsockopt(m_socket.get(), SOL_SOCKET, SO_LINGER, &lng, sizeof(lng));

	::shutdown(m_socket.get(), SHUT_RDWR);
}

const IpPort &TcpSessionBase::get_remote_info() const {
	PROFILE_ME;

	fetch_peer_info();
	return m_peer_info.remote;
}
const IpPort &TcpSessionBase::get_local_info() const {
	PROFILE_ME;

	fetch_peer_info();
	return m_peer_info.local;
}
IpPort TcpSessionBase::get_remote_info_nothrow() const NOEXCEPT
try {
	PROFILE_ME;

	return get_remote_info();
} catch(std::exception &e){
	LOG_POSEIDON_DEBUG("std::exception thrown: what = ", e.what());
	return IpPort(sslit("<unknown>"), 0);
}
IpPort TcpSessionBase::get_local_info_nothrow() const NOEXCEPT
try {
	PROFILE_ME;

	return get_local_info();
} catch(std::exception &e){
	LOG_POSEIDON_DEBUG("std::exception thrown: what = ", e.what());
	return IpPort(sslit("<unknown>"), 0);
}

void TcpSessionBase::set_timeout(boost::uint64_t timeout){
	PROFILE_ME;

	const Mutex::UniqueLock lock(m_timer_mutex);
	if(timeout == 0){
		m_shutdown_timer.reset();
	} else {
		if(!m_shutdown_timer){
			const AUTO(now, get_fast_mono_clock());
			m_shutdown_timer = TimerDaemon::register_low_level_absolute_timer(saturated_add(now, timeout), timeout,
				boost::bind(&shutdown_timer_proc, virtual_weak_from_this<TcpSessionBase>(), _2));
		}
		m_shutdown_time = get_fast_mono_clock() + timeout;
	}
}

void TcpSessionBase::set_no_delay(bool enabled){
	PROFILE_ME;

	const int val = enabled;
	if(::setsockopt(m_socket.get(), IPPROTO_TCP, TCP_NODELAY, &val, sizeof(val)) != 0){
		const int err_code = errno;
		LOG_POSEIDON_WARNING("Error setting TCP socket option: err_code = ", err_code);
		DEBUG_THROW(SystemException, err_code);
	}
}

bool TcpSessionBase::is_throttled() const {
	return atomic_load(m_throttled, ATOMIC_CONSUME);
}
void TcpSessionBase::set_throttled(bool throttled){
	atomic_store(m_throttled, throttled, ATOMIC_RELEASE);
}

}
