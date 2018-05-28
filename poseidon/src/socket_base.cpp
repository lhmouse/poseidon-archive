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

Socket_base::Delayed_shutdown_guard::Delayed_shutdown_guard(boost::weak_ptr<Socket_base> weak)
	: m_weak(STD_MOVE(weak))
{
	const AUTO(socket, m_weak.lock());
	if(!socket){
		return;
	}
	atomic_add(socket->m_delayed_shutdown_guard_count, 1, memory_order_relaxed);
}
Socket_base::Delayed_shutdown_guard::~Delayed_shutdown_guard(){
	const AUTO(socket, m_weak.lock());
	if(!socket){
		return;
	}
	if(atomic_sub(socket->m_delayed_shutdown_guard_count, 1, memory_order_relaxed) == 0){
		if(atomic_load(socket->m_shutdown_write, memory_order_acquire)){
			atomic_store(socket->m_really_shutdown_write, true, memory_order_release);
			Epoll_daemon::mark_socket_writable(socket.get());
		}
	}
}

Socket_base::Socket_base(Move<Unique_file> socket)
	: m_socket(STD_MOVE(socket)), m_creation_time(get_utc_time())
	, m_shutdown_read(false), m_shutdown_write(false), m_really_shutdown_write(false)
	, m_throttled(false), m_timed_out(false), m_delayed_shutdown_guard_count(0)
{
	//
}
Socket_base::~Socket_base(){
	// This FD may have been dup()'d.
	::shutdown(get_fd(), SHUT_RDWR);
}

void Socket_base::fetch_remote_info_unlocked() const {
	POSEIDON_PROFILE_ME;

	if(is_listening()){
		m_remote_info = listening_ip_port();
		return;
	}
	::sockaddr_storage sa;
	::socklen_t salen = sizeof(sa);
	if(::getpeername(get_fd(), static_cast< ::sockaddr *>(static_cast<void *>(&sa)), &salen) != 0){
		return;
	}
	const Sock_addr sock_addr(&sa, salen);
	m_remote_info = sock_addr;
}
void Socket_base::fetch_local_info_unlocked() const {
	POSEIDON_PROFILE_ME;

	::sockaddr_storage sa;
	::socklen_t salen = sizeof(sa);
	if(::getsockname(get_fd(), static_cast< ::sockaddr *>(static_cast<void *>(&sa)), &salen) != 0){
		return;
	}
	const Sock_addr sock_addr(&sa, salen);
	m_local_info = sock_addr;
	m_ipv6 = sock_addr.is_ipv6();
}

bool Socket_base::should_really_shutdown_write() const NOEXCEPT {
	return atomic_load(m_really_shutdown_write, memory_order_acquire);
}
void Socket_base::set_timed_out() NOEXCEPT {
	atomic_store(m_timed_out, true, memory_order_release);
}

bool Socket_base::is_listening() const {
	POSEIDON_PROFILE_ME;

	int val;
	::socklen_t len = sizeof(val);
	POSEIDON_THROW_UNLESS(::getsockopt(get_fd(), SOL_SOCKET, SO_ACCEPTCONN, &val, &len) == 0, System_exception);
	return (len >= sizeof(int)) && (val != 0);
}

bool Socket_base::has_been_shutdown_read() const NOEXCEPT {
	return atomic_load(m_shutdown_read, memory_order_acquire);
}
bool Socket_base::shutdown_read() NOEXCEPT {
	POSEIDON_PROFILE_ME;

	bool was_shutdown_read = atomic_load(m_shutdown_read, memory_order_acquire);
	if(!was_shutdown_read){
		was_shutdown_read = atomic_exchange(m_shutdown_read, true, memory_order_acq_rel);
	}
	if(was_shutdown_read){
		return false;
	}
	::shutdown(get_fd(), SHUT_RD);
	return true;
}
bool Socket_base::has_been_shutdown_write() const NOEXCEPT {
	return atomic_load(m_shutdown_write, memory_order_acquire);
}
bool Socket_base::shutdown_write() NOEXCEPT {
	POSEIDON_PROFILE_ME;

	bool was_shutdown_write = atomic_load(m_shutdown_write, memory_order_acquire);
	if(!was_shutdown_write){
		was_shutdown_write = atomic_exchange(m_shutdown_write, true, memory_order_acq_rel);
	}
	if(was_shutdown_write){
		return false;
	}
	const Delayed_shutdown_guard guard(virtual_shared_from_this<Socket_base>());
	return true;
}
void Socket_base::mark_shutdown() NOEXCEPT {
	POSEIDON_PROFILE_ME;

	atomic_store(m_shutdown_read, true, memory_order_release);
	atomic_store(m_shutdown_write, true, memory_order_release);
}
void Socket_base::force_shutdown() NOEXCEPT {
	POSEIDON_PROFILE_ME;

	bool was_shutdown_read = atomic_load(m_shutdown_read, memory_order_acquire);
	if(!was_shutdown_read){
		was_shutdown_read = atomic_exchange(m_shutdown_read, true, memory_order_acq_rel);
	}
	bool was_shutdown_write = atomic_load(m_shutdown_write, memory_order_acquire);
	if(!was_shutdown_write){
		was_shutdown_write = atomic_exchange(m_shutdown_write, true, memory_order_acq_rel);
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
			POSEIDON_LOG_WARNING("::setsockopt() failed, errno was ", err_code);
		}
	}
	::shutdown(get_fd(), SHUT_RDWR);
}

bool Socket_base::is_throttled() const {
	return atomic_load(m_throttled, memory_order_acquire);
}
void Socket_base::set_throttled(bool throttled){
	atomic_store(m_throttled, throttled, memory_order_release);
}

bool Socket_base::did_time_out() const NOEXCEPT {
	return atomic_load(m_timed_out, memory_order_acquire);
}

const Ip_port & Socket_base::get_remote_info() const NOEXCEPT
try {
	POSEIDON_PROFILE_ME;

	const std::lock_guard<std::mutex> lock(m_info_mutex);
	if(!m_remote_info){
		fetch_remote_info_unlocked();
	}
	if(!m_remote_info){
		return unknown_ip_port();
	}
	return m_remote_info.get();
} catch(std::exception &e){
	POSEIDON_LOG_ERROR("std::exception thrown: what = ", e.what());
	return unknown_ip_port();
}
const Ip_port & Socket_base::get_local_info() const NOEXCEPT
try {
	POSEIDON_PROFILE_ME;

	const std::lock_guard<std::mutex> lock(m_info_mutex);
	if(!m_local_info){
		fetch_local_info_unlocked();
	}
	if(!m_local_info){
		return unknown_ip_port();
	}
	return m_local_info.get();
} catch(std::exception &e){
	POSEIDON_LOG_ERROR("std::exception thrown: what = ", e.what());
	return unknown_ip_port();
}
bool Socket_base::is_using_ipv6() const NOEXCEPT
try {
	POSEIDON_PROFILE_ME;

	const std::lock_guard<std::mutex> lock(m_info_mutex);
	if(!m_ipv6){
		fetch_local_info_unlocked();
	}
	if(!m_ipv6){
		return false;
	}
	return m_ipv6.get();
} catch(std::exception &e){
	POSEIDON_LOG_ERROR("std::exception thrown: what = ", e.what());
	return false;
}

int Socket_base::poll_read_and_process(unsigned char */*hint_buffer*/, std::size_t /*hint_capacity*/, bool /*readable*/){
	return EWOULDBLOCK;
}
int Socket_base::poll_write(std::unique_lock<std::mutex> &/*write_lock*/, unsigned char */*hint_buffer*/, std::size_t /*hint_capacity*/, bool /*writable*/){
	return EWOULDBLOCK;
}
void Socket_base::on_close(int /*err_code*/){
	//
}

}
