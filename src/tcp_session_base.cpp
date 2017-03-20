// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2017, LH_Mouse. All wrongs reserved.

#include "precompiled.hpp"
#include "tcp_session_base.hpp"
#include "ssl_filter_base.hpp"
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include "singletons/epoll_daemon.hpp"
#include "log.hpp"
#include "system_exception.hpp"
#include "profiler.hpp"
#include "atomic.hpp"
#include "checked_arithmetic.hpp"
#include "singletons/timer_daemon.hpp"
#include "time.hpp"

namespace Poseidon {

void TcpSessionBase::shutdown_timer_proc(const boost::weak_ptr<TcpSessionBase> &weak, boost::uint64_t now){
	const AUTO(session, weak.lock());
	if(!session){
		return;
	}
	if(now < session->m_shutdown_time){
		return;
	}

	std::size_t send_buffer_size;
	{
		const Poseidon::Mutex::UniqueLock lock(session->m_send_mutex);
		send_buffer_size = session->m_send_buffer.size();
	}
	if(send_buffer_size > 0){
		LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_DEBUG,
			"Shutdown pending: remote = ", session->get_remote_info(), ", send_buffer_size = ", send_buffer_size);
		return;
	}
	LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_DEBUG,
		"Connection timed out: remote = ", session->get_remote_info());
	session->set_timed_out();
	session->force_shutdown();
}

TcpSessionBase::TcpSessionBase(UniqueFile socket)
	: SocketBase(STD_MOVE(socket)), SessionBase()
	, m_connected(false), m_connected_notified(false), m_read_hup_notified(false)
{
}
TcpSessionBase::~TcpSessionBase(){
}

void TcpSessionBase::init_ssl(Move<boost::scoped_ptr<SslFilterBase> > ssl_filter){
	swap(m_ssl_filter, ssl_filter);
}

int TcpSessionBase::poll_read_and_process(bool readable){
	PROFILE_ME;

	(void)readable;

	std::string data;
	try {
		data.resize(4096);

		::ssize_t result;
		if(m_ssl_filter){
			result = m_ssl_filter->recv(&data[0], data.size());
		} else {
			result = ::recv(get_fd(), &data[0], data.size(), MSG_NOSIGNAL | MSG_DONTWAIT);
		}
		if(result < 0){
			return errno;
		}
		data.erase(data.begin() + result, data.end());
		LOG_POSEIDON_TRACE("Read ", result, " byte(s) from ", get_remote_info());
	} catch(std::exception &e){
		LOG_POSEIDON_ERROR("std::exception thrown: what = ", e.what());
		force_shutdown();
		return EPIPE;
	}

	try {
		if(data.empty()){
			if(!m_read_hup_notified){
				on_read_hup();
				m_read_hup_notified = true;
			}
			return EWOULDBLOCK;
		}
		on_receive(StreamBuffer(data));
	} catch(std::exception &e){
		LOG_POSEIDON_ERROR("std::exception thrown: what = ", e.what());
		force_shutdown();
		return EPIPE;
	} catch(...){
		LOG_POSEIDON_ERROR("Unknown exception thrown.");
		force_shutdown();
		return EPIPE;
	}
	return 0;
}
int TcpSessionBase::poll_write(Mutex::UniqueLock &write_lock, bool writeable){
	PROFILE_ME;

	assert(!write_lock);

	try {
		if(writeable){
			atomic_store(m_connected, true, ATOMIC_RELEASE);
			if(!m_connected_notified){
				on_connect();
				m_connected_notified = true;
			}
		}
	} catch(std::exception &e){
		LOG_POSEIDON_ERROR("std::exception thrown: what = ", e.what());
		force_shutdown();
		return EPIPE;
	} catch(...){
		LOG_POSEIDON_ERROR("Unknown exception thrown.");
		force_shutdown();
		return EPIPE;
	}

	std::string data;
	try {
		data.resize(4096);

		Poseidon::Mutex::UniqueLock lock(m_send_mutex);
		const std::size_t avail = m_send_buffer.peek(&data[0], data.size());
		if(avail == 0){
			if(should_really_shutdown_write()){
				if(m_ssl_filter){
					m_ssl_filter->send_fin();
				} else {
					::shutdown(get_fd(), SHUT_WR);
				}
			}
			return EWOULDBLOCK;
		}
		lock.unlock();
		data.erase(data.begin() + static_cast<std::ptrdiff_t>(avail), data.end());

		::ssize_t result;
		if(m_ssl_filter){
			result = m_ssl_filter->send(&data[0], data.size());
		} else {
			result = ::send(get_fd(), &data[0], data.size(), MSG_NOSIGNAL | MSG_DONTWAIT);
		}
		if(result < 0){
			return errno;
		}
		lock.lock();
		m_send_buffer.discard(static_cast<std::size_t>(result));
		LOG_POSEIDON_TRACE("Wrote ", result, " byte(s) to ", get_remote_info());
		swap(write_lock, lock);
		if(m_send_buffer.empty()){
			return EWOULDBLOCK;
		}
	} catch(std::exception &e){
		LOG_POSEIDON_ERROR("std::exception thrown: what = ", e.what());
		force_shutdown();
		return EPIPE;
	}
	return 0;
}

void TcpSessionBase::on_connect(){
	LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_DEBUG,
		"TCP connection established: local = ", get_local_info(), ", remote = ", get_remote_info());
}
void TcpSessionBase::on_read_hup() NOEXCEPT {
	LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_DEBUG,
		"TCP connection read hung up: local = ", get_local_info(), ", remote = ", get_remote_info());
	SocketBase::shutdown_read();
}
void TcpSessionBase::on_close(int err_code) NOEXCEPT {
	LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_DEBUG,
		"TCP connection closed: local = ", get_local_info(), ", remote = ", get_remote_info(), ", err_code = ", err_code);
	SocketBase::shutdown_read();
	SocketBase::shutdown_write();
	atomic_store(m_connected, false, ATOMIC_RELEASE);
}

bool TcpSessionBase::is_throttled() const {
	const Mutex::UniqueLock lock(m_send_mutex);
	if(m_send_buffer.size() >= 65536){
		return true;
	}
	return SocketBase::is_throttled();
}
bool TcpSessionBase::is_connected() const NOEXCEPT {
	return atomic_load(m_connected, ATOMIC_CONSUME);
}

void TcpSessionBase::set_no_delay(bool enabled){
	PROFILE_ME;

	const int val = enabled;
	if(::setsockopt(get_fd(), IPPROTO_TCP, TCP_NODELAY, &val, sizeof(val)) != 0){
		DEBUG_THROW(SystemException);
	}
}
void TcpSessionBase::set_timeout(boost::uint64_t timeout){
	PROFILE_ME;

	if(timeout == 0){
		const Mutex::UniqueLock lock(m_shutdown_mutex);
		m_shutdown_timer.reset();
	} else {
		const AUTO(now, get_fast_mono_clock());
		const Mutex::UniqueLock lock(m_shutdown_mutex);
		if(!m_shutdown_timer){
			m_shutdown_timer = TimerDaemon::register_low_level_absolute_timer(now + 5000, 5000,
				boost::bind(&shutdown_timer_proc, virtual_weak_from_this<TcpSessionBase>(), _2));
		}
		m_shutdown_time = saturated_add(now, timeout);
	}
}

bool TcpSessionBase::send(StreamBuffer buffer){
	PROFILE_ME;

	if(has_been_shutdown_write()){
		LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_DEBUG,
			"TCP socket has been shut down for writing: local = ", get_local_info(), ", remote = ", get_remote_info());
		return false;
	}

	const Mutex::UniqueLock lock(m_send_mutex);
	m_send_buffer.splice(buffer);
	EpollDaemon::mark_socket_writeable(get_fd());
	return true;
}

}
