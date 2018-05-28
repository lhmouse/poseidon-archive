// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2018, LH_Mouse. All wrongs reserved.

#include "precompiled.hpp"
#include "tcp_session_base.hpp"
#include "ssl_filter.hpp"
#include "singletons/epoll_daemon.hpp"
#include "singletons/main_config.hpp"
#include "log.hpp"
#include "system_exception.hpp"
#include "profiler.hpp"
#include "atomic.hpp"
#include "checked_arithmetic.hpp"
#include "singletons/timer_daemon.hpp"
#include "time.hpp"
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netinet/tcp.h>

namespace Poseidon {

void Tcp_session_base::shutdown_timer_proc(const boost::weak_ptr<Tcp_session_base> &weak, std::uint64_t now){
	POSEIDON_PROFILE_ME;

	const AUTO(session, weak.lock());
	if(!session){
		return;
	}

	try {
		session->on_shutdown_timer(now);
	} catch(std::exception &e){
		POSEIDON_LOG_WARNING("std::exception thrown: what = ", e.what());
		session->force_shutdown();
	}
}

Tcp_session_base::Tcp_session_base(Move<Unique_file> socket)
	: Socket_base(STD_MOVE(socket)), Session_base()
	, m_connected_notified(false), m_read_hup_notified(false)
	, m_shutdown_time(-1ull), m_last_use_time(-1ull)
{
	//
}
Tcp_session_base::~Tcp_session_base(){
	//
}

void Tcp_session_base::init_ssl(boost::scoped_ptr<Ssl_filter> &ssl_filter){
	POSEIDON_THROW_ASSERT(!m_ssl_filter);
	swap(m_ssl_filter, ssl_filter);
}
void Tcp_session_base::create_shutdown_timer(){
	POSEIDON_PROFILE_ME;

	const std::lock_guard<std::mutex> lock(m_shutdown_mutex);
	if(m_shutdown_timer){
		return;
	}
	const AUTO(period, Main_config::get<std::uint64_t>("tcp_shutdown_timer_period", 15000));
	m_shutdown_timer = Timer_daemon::register_low_level_timer(period, period, std::bind(&shutdown_timer_proc, virtual_weak_from_this<Tcp_session_base>(), std::placeholders::_2));
}

int Tcp_session_base::poll_read_and_process(unsigned char *hint_buffer, std::size_t hint_capacity, bool /*readable*/){
	POSEIDON_PROFILE_ME;

	Stream_buffer data;
	try {
		::ssize_t result;
		if(m_ssl_filter){
			result = m_ssl_filter->recv(hint_buffer, hint_capacity);
		} else {
			result = ::recv(get_fd(), hint_buffer, hint_capacity, MSG_NOSIGNAL | MSG_DONTWAIT);
		}
		if(result < 0){
			return errno;
		}
		data.put(hint_buffer, static_cast<std::size_t>(result));
		POSEIDON_LOG_TRACE("Read ", result, " byte(s) from ", get_remote_info());

		const AUTO(now, get_fast_mono_clock());
		atomic_store(m_last_use_time, now, memory_order_release);
		create_shutdown_timer();

		if(data.empty() && !m_read_hup_notified){
			POSEIDON_LOG(Logger::special_major | Logger::level_debug, "TCP connection read hung up: local = ", get_local_info(), ", remote = ", get_remote_info());
			shutdown_read();
			on_read_hup();
			m_read_hup_notified = true;
		}
		if(data.empty()){
			return EWOULDBLOCK;
		}
		on_receive(STD_MOVE(data));
	} catch(std::exception &e){
		POSEIDON_LOG_ERROR("std::exception thrown: what = ", e.what());
		force_shutdown();
		return EPIPE;
	} catch(...){
		POSEIDON_LOG_ERROR("Unknown exception thrown.");
		force_shutdown();
		return EPIPE;
	}
	return 0;
}
int Tcp_session_base::poll_write(std::unique_lock<std::mutex> &write_lock, unsigned char *hint_buffer, std::size_t hint_capacity, bool writable){
	POSEIDON_PROFILE_ME;

	assert(!write_lock);

	Stream_buffer data;
	try {
		if(writable && !m_connected_notified){
			POSEIDON_LOG(Logger::special_major | Logger::level_debug, "TCP connection established: local = ", get_local_info(), ", remote = ", get_remote_info());
			on_connect();
			m_connected_notified = true;
		}

		std::unique_lock<std::mutex> lock(m_send_mutex);
		const std::size_t avail = m_send_buffer.peek(hint_buffer, hint_capacity);
		if(avail == 0){
_check_shutdown:
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

		::ssize_t result;
		if(m_ssl_filter){
			result = m_ssl_filter->send(hint_buffer, avail);
		} else {
			result = ::send(get_fd(), hint_buffer, avail, MSG_NOSIGNAL | MSG_DONTWAIT);
		}
		if(result < 0){
			return errno;
		}
		POSEIDON_LOG_TRACE("Wrote ", result, " byte(s) to ", get_remote_info());

		const AUTO(now, get_fast_mono_clock());
		atomic_store(m_last_use_time, now, memory_order_release);
		create_shutdown_timer();

		lock.lock();
		m_send_buffer.discard(static_cast<std::size_t>(result));
		swap(write_lock, lock);
		if(m_send_buffer.empty()){
			goto _check_shutdown;
		}
	} catch(std::exception &e){
		POSEIDON_LOG_ERROR("std::exception thrown: what = ", e.what());
		force_shutdown();
		return EPIPE;
	} catch(...){
		POSEIDON_LOG_ERROR("Unknown exception thrown.");
		force_shutdown();
		return EPIPE;
	}
	return 0;
}

void Tcp_session_base::on_shutdown_timer(std::uint64_t now){
	POSEIDON_PROFILE_ME;

	const AUTO(shutdown_time, atomic_load(m_shutdown_time, memory_order_consume));
	if(shutdown_time < now){
		std::size_t send_buffer_size;
		{
			const std::lock_guard<std::mutex> lock(m_send_mutex);
			send_buffer_size = m_send_buffer.size();
		}
		if(send_buffer_size == 0){
			POSEIDON_LOG(Logger::special_major | Logger::level_debug, "Connection closed due to inactivity: remote = ", get_remote_info());
force_time_out:
			set_timed_out();
			force_shutdown();
			return;
		}
		POSEIDON_LOG(Logger::special_major | Logger::level_debug, "Shutdown pending: remote = ", get_remote_info(), ", send_buffer_size = ", send_buffer_size);
	}

	const AUTO(last_use_time, atomic_load(m_last_use_time, memory_order_consume));
	const AUTO(tcp_response_timeout, Main_config::get<std::uint64_t>("tcp_response_timeout", 30000));
	if(saturated_sub(now, last_use_time) > tcp_response_timeout){
		POSEIDON_LOG(Logger::special_major | Logger::level_debug, "The connection seems dead: remote = ", get_remote_info());
		goto force_time_out;
	}
}

bool Tcp_session_base::has_been_shutdown_read() const NOEXCEPT {
	return Socket_base::has_been_shutdown_read();
}
bool Tcp_session_base::has_been_shutdown_write() const NOEXCEPT {
	return Socket_base::has_been_shutdown_write();
}
bool Tcp_session_base::shutdown_read() NOEXCEPT {
	return Socket_base::shutdown_read();
}
bool Tcp_session_base::shutdown_write() NOEXCEPT {
	return Socket_base::shutdown_write();
}
void Tcp_session_base::force_shutdown() NOEXCEPT {
	Socket_base::force_shutdown();
}

bool Tcp_session_base::is_using_ssl() const {
	return !!m_ssl_filter;
}
bool Tcp_session_base::is_throttled() const {
	const std::lock_guard<std::mutex> lock(m_send_mutex);
	if(m_send_buffer.size() >= 65536){
		return true;
	}
	return Socket_base::is_throttled();
}

void Tcp_session_base::set_no_delay(bool enabled){
	POSEIDON_PROFILE_ME;

	const int val = enabled;
	POSEIDON_THROW_UNLESS(::setsockopt(get_fd(), IPPROTO_TCP, TCP_NODELAY, &val, sizeof(val)) == 0, System_exception);
}
void Tcp_session_base::set_timeout(std::uint64_t timeout){
	POSEIDON_PROFILE_ME;

	const AUTO(now, get_fast_mono_clock());
	atomic_store(m_shutdown_time, saturated_add(now, timeout), memory_order_release);
	create_shutdown_timer();
}

bool Tcp_session_base::send(Stream_buffer buffer){
	POSEIDON_PROFILE_ME;

	if(has_been_shutdown_write()){
		POSEIDON_LOG(Logger::special_major | Logger::level_debug, "TCP socket has been shut down for writing: local = ", get_local_info(), ", remote = ", get_remote_info());
		return false;
	}

	const std::lock_guard<std::mutex> lock(m_send_mutex);
	m_send_buffer.splice(buffer);
	Epoll_daemon::mark_socket_writable(this);
	return true;
}

}
