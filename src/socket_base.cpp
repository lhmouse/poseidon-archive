// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2017, LH_Mouse. All wrongs reserved.

#include "precompiled.hpp"
#include "socket_base.hpp"
#include <sys/types.h>
#include <sys/socket.h>
#include <fcntl.h>
#include <errno.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include "log.hpp"
#include "profiler.hpp"
#include "atomic.hpp"
#include "time.hpp"
#include "system_exception.hpp"
#include "ip_port.hpp"
#include "sock_addr.hpp"
#include "singletons/epoll_daemon.hpp"

namespace Poseidon {

SocketBase::DelayedShutdownGuard::DelayedShutdownGuard(boost::weak_ptr<SocketBase> weak)
	: m_weak(STD_MOVE(weak))
{
	const AUTO(socket, m_weak.lock());
	if(!socket){
		return;
	}
	atomic_add(socket->m_delayed_shutdown_guard_count, 1, ATOMIC_RELAXED);
}
SocketBase::DelayedShutdownGuard::~DelayedShutdownGuard(){
	const AUTO(socket, m_weak.lock());
	if(!socket){
		return;
	}
	if(atomic_sub(socket->m_delayed_shutdown_guard_count, 1, ATOMIC_RELAXED) == 0){
		if(atomic_load(socket->m_shutdown_write, ATOMIC_CONSUME)){
			atomic_store(socket->m_really_shutdown_write, true, ATOMIC_RELEASE);
			const bool pending = EpollDaemon::mark_socket_writeable(socket->get_fd());
			if(!pending){
				::shutdown(socket->get_fd(), SHUT_WR);
			}
		}
	}
}

SocketBase::SocketBase(UniqueFile socket)
	: m_socket(STD_MOVE(socket)), m_creation_time(get_fast_mono_clock())
	, m_shutdown_read(false), m_shutdown_write(false), m_really_shutdown_write(false)
	, m_throttled(false), m_timed_out(false), m_delayed_shutdown_guard_count(0)
{
	const int flags = ::fcntl(m_socket.get(), F_GETFL);
	if(flags == -1){
		DEBUG_THROW(SystemException);
	}
	if(::fcntl(m_socket.get(), F_SETFL, flags | O_NONBLOCK) != 0){
		DEBUG_THROW(SystemException);
	}
}
SocketBase::~SocketBase(){
}

bool SocketBase::should_really_shutdown_write() const NOEXCEPT {
	return atomic_load(m_really_shutdown_write, ATOMIC_CONSUME);
}
void SocketBase::set_timed_out() NOEXCEPT {
	atomic_store(m_timed_out, true, ATOMIC_RELEASE);
}

bool SocketBase::has_been_shutdown_read() const NOEXCEPT {
	return atomic_load(m_shutdown_read, ATOMIC_CONSUME);
}
bool SocketBase::shutdown_read() NOEXCEPT {
	PROFILE_ME;

	bool ret = !atomic_load(m_shutdown_read, ATOMIC_CONSUME);
	if(ret){
		ret = !atomic_exchange(m_shutdown_read, true, ATOMIC_ACQ_REL);
		::shutdown(get_fd(), SHUT_RD);
	}
	return ret;
}
bool SocketBase::has_been_shutdown_write() const NOEXCEPT {
	return atomic_load(m_shutdown_write, ATOMIC_CONSUME);
}
bool SocketBase::shutdown_write() NOEXCEPT {
	PROFILE_ME;

	bool ret = !atomic_load(m_shutdown_write, ATOMIC_CONSUME);
	if(ret){
		ret = !atomic_exchange(m_shutdown_write, true, ATOMIC_ACQ_REL);
		const DelayedShutdownGuard guard(virtual_shared_from_this<SocketBase>());
	}
	return ret;
}
void SocketBase::force_shutdown() NOEXCEPT {
	PROFILE_ME;

	atomic_store(m_shutdown_read, true, ATOMIC_RELEASE);
	atomic_store(m_shutdown_write, true, ATOMIC_RELEASE);

	::linger lng;
	lng.l_onoff = 1;
	lng.l_linger = 0;
	::setsockopt(get_fd(), SOL_SOCKET, SO_LINGER, &lng, sizeof(lng));

	::shutdown(get_fd(), SHUT_RDWR);
}

bool SocketBase::is_throttled() const {
	return atomic_load(m_throttled, ATOMIC_CONSUME);
}
void SocketBase::set_throttled(bool throttled){
	atomic_store(m_throttled, throttled, ATOMIC_RELEASE);
}

bool SocketBase::was_timed_out() const NOEXCEPT {
	return atomic_load(m_timed_out, ATOMIC_CONSUME);
}

const IpPort &SocketBase::get_remote_info() const NOEXCEPT
try {
	PROFILE_ME;

	const Mutex::UniqueLock lock(m_info_mutex);
	if(m_remote_info.port() == 0){
		::sockaddr_storage sa;
		::socklen_t salen = sizeof(sa);
		if(::getpeername(get_fd(), static_cast< ::sockaddr *>(static_cast<void *>(&sa)), &salen) != 0){
			DEBUG_THROW(SystemException);
		}
		m_remote_info = SockAddr(&sa, salen);
	}
	return m_remote_info;
} catch(std::exception &e){
	LOG_POSEIDON_DEBUG("std::exception thrown: what = ", e.what());
	return unknown_ip_port();
}
const IpPort &SocketBase::get_local_info() const NOEXCEPT
try {
	PROFILE_ME;

	const Mutex::UniqueLock lock(m_info_mutex);
	if(m_local_info.port() == 0){
		::sockaddr_storage sa;
		::socklen_t salen = sizeof(sa);
		if(::getsockname(get_fd(), static_cast< ::sockaddr *>(static_cast<void *>(&sa)), &salen) != 0){
			DEBUG_THROW(SystemException);
		}
		m_local_info = SockAddr(&sa, salen);
	}
	return m_local_info;
} catch(std::exception &e){
	LOG_POSEIDON_DEBUG("std::exception thrown: what = ", e.what());
	return unknown_ip_port();
}

int SocketBase::poll_read_and_process(bool readable){
	(void)readable;
	return EWOULDBLOCK;
}
int SocketBase::poll_write(Mutex::UniqueLock &write_lock, bool writeable){
	(void)write_lock;
	(void)writeable;
	return EWOULDBLOCK;
}
void SocketBase::on_close(int err_code) NOEXCEPT {
	(void)err_code;
}

}
