// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2018, LH_Mouse. All wrongs reserved.

#include "precompiled.hpp"
#include "socket_base.hpp"
#include <sys/types.h>
#include <sys/socket.h>
#include <fcntl.h>
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
	atomic_add(socket->m_delayed_shutdown_guard_count, 1, memorder_relaxed);
}
SocketBase::DelayedShutdownGuard::~DelayedShutdownGuard(){
	const AUTO(socket, m_weak.lock());
	if(!socket){
		return;
	}
	if(atomic_sub(socket->m_delayed_shutdown_guard_count, 1, memorder_relaxed) == 0){
		if(atomic_load(socket->m_shutdown_write, memorder_acquire)){
			atomic_store(socket->m_really_shutdown_write, true, memorder_release);
			EpollDaemon::mark_socket_writeable(socket.get());
		}
	}
}

SocketBase::SocketBase(Move<UniqueFile> socket)
	: m_socket(STD_MOVE(socket)), m_creation_time(get_utc_time())
	, m_shutdown_read(false), m_shutdown_write(false), m_really_shutdown_write(false)
	, m_throttled(false), m_timed_out(false), m_delayed_shutdown_guard_count(0)
{
	//
}
SocketBase::~SocketBase(){
	// This FD may have been dup()'d.
	::shutdown(get_fd(), SHUT_RDWR);
}

bool SocketBase::should_really_shutdown_write() const NOEXCEPT {
	return atomic_load(m_really_shutdown_write, memorder_acquire);
}
void SocketBase::set_timed_out() NOEXCEPT {
	atomic_store(m_timed_out, true, memorder_release);
}

bool SocketBase::is_listening() const {
	PROFILE_ME;

	int val;
	::socklen_t len = sizeof(val);
	DEBUG_THROW_UNLESS(::getsockopt(get_fd(), SOL_SOCKET, SO_ACCEPTCONN, &val, &len) == 0, SystemException);
	return (len >= sizeof(int)) && (val != 0);
}

bool SocketBase::has_been_shutdown_read() const NOEXCEPT {
	return atomic_load(m_shutdown_read, memorder_acquire);
}
bool SocketBase::shutdown_read() NOEXCEPT {
	PROFILE_ME;

	bool was_shutdown_read = atomic_load(m_shutdown_read, memorder_acquire);
	if(!was_shutdown_read){
		was_shutdown_read = atomic_exchange(m_shutdown_read, true, memorder_acq_rel);
	}
	if(was_shutdown_read){
		return false;
	}
	::shutdown(get_fd(), SHUT_RD);
	return true;
}
bool SocketBase::has_been_shutdown_write() const NOEXCEPT {
	return atomic_load(m_shutdown_write, memorder_acquire);
}
bool SocketBase::shutdown_write() NOEXCEPT {
	PROFILE_ME;

	bool was_shutdown_write = atomic_load(m_shutdown_write, memorder_acquire);
	if(!was_shutdown_write){
		was_shutdown_write = atomic_exchange(m_shutdown_write, true, memorder_acq_rel);
	}
	if(was_shutdown_write){
		return false;
	}
	const DelayedShutdownGuard guard(virtual_shared_from_this<SocketBase>());
	return true;
}
void SocketBase::mark_shutdown() NOEXCEPT {
	PROFILE_ME;

	atomic_store(m_shutdown_read, true, memorder_release);
	atomic_store(m_shutdown_write, true, memorder_release);
}
void SocketBase::force_shutdown() NOEXCEPT {
	PROFILE_ME;

	bool was_shutdown_read = atomic_load(m_shutdown_read, memorder_acquire);
	if(!was_shutdown_read){
		was_shutdown_read = atomic_exchange(m_shutdown_read, true, memorder_acq_rel);
	}
	bool was_shutdown_write = atomic_load(m_shutdown_write, memorder_acquire);
	if(!was_shutdown_write){
		was_shutdown_write = atomic_exchange(m_shutdown_write, true, memorder_acq_rel);
	}
	if(was_shutdown_read && was_shutdown_write){
		return;
	}
	if(!was_shutdown_write){
		::linger lng;
		lng.l_onoff = 1;
		lng.l_linger = 0;
		if(::setsockopt(get_fd(), SOL_SOCKET, SO_LINGER, &lng, sizeof(lng)) != 0){
			const int err_code = errno;
			LOG_POSEIDON_WARNING("::setsockopt() failed, errno was ", err_code);
		}
	}
	::shutdown(get_fd(), SHUT_RDWR);
}

bool SocketBase::is_throttled() const {
	return atomic_load(m_throttled, memorder_acquire);
}
void SocketBase::set_throttled(bool throttled){
	atomic_store(m_throttled, throttled, memorder_release);
}

bool SocketBase::did_time_out() const NOEXCEPT {
	return atomic_load(m_timed_out, memorder_acquire);
}

const IpPort &SocketBase::get_remote_info() const NOEXCEPT
try {
	PROFILE_ME;

	const Mutex::UniqueLock lock(m_info_mutex);
	if(m_remote_info.port() != 0){
		return m_remote_info;
	}
	if(is_listening()){
		return m_remote_info = listening_ip_port();
	}
	::sockaddr_storage sa;
	::socklen_t salen = sizeof(sa);
	if(::getpeername(get_fd(), static_cast< ::sockaddr *>(static_cast<void *>(&sa)), &salen) != 0){
		return unknown_ip_port();
	}
	return m_remote_info = SockAddr(&sa, salen);
} catch(std::exception &e){
	LOG_POSEIDON_DEBUG("std::exception thrown: what = ", e.what());
	return unknown_ip_port();
}
const IpPort &SocketBase::get_local_info() const NOEXCEPT
try {
	PROFILE_ME;

	const Mutex::UniqueLock lock(m_info_mutex);
	if(m_local_info.port() != 0){
		return m_local_info;
	}
	::sockaddr_storage sa;
	::socklen_t salen = sizeof(sa);
	if(::getsockname(get_fd(), static_cast< ::sockaddr *>(static_cast<void *>(&sa)), &salen) != 0){
		return unknown_ip_port();
	}
	return m_local_info = SockAddr(&sa, salen);
} catch(std::exception &e){
	LOG_POSEIDON_DEBUG("std::exception thrown: what = ", e.what());
	return unknown_ip_port();
}

int SocketBase::poll_read_and_process(unsigned char *hint_buffer, std::size_t hint_capacity, bool readable){
	(void)hint_buffer;
	(void)hint_capacity;
	(void)readable;

	return EWOULDBLOCK;
}
int SocketBase::poll_write(Mutex::UniqueLock &write_lock, unsigned char *hint_buffer, std::size_t hint_capacity, bool writeable){
	(void)write_lock;
	(void)hint_buffer;
	(void)hint_capacity;
	(void)writeable;

	return EWOULDBLOCK;
}
void SocketBase::on_close(int err_code){
	(void)err_code;
}

}
