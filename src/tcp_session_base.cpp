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
#include "singletons/main_config.hpp"
#include "log.hpp"
#include "system_exception.hpp"
#include "profiler.hpp"
#include "atomic.hpp"
#include "checked_arithmetic.hpp"
#include "singletons/timer_daemon.hpp"
#include "time.hpp"

namespace Poseidon {

void TcpSessionBase::shutdown_timer_proc(const boost::weak_ptr<TcpSessionBase> &weak, boost::uint64_t now){
	PROFILE_ME;

	const AUTO(session, weak.lock());
	if(!session){
		return;
	}

	const AUTO(shutdown_time, atomic_load(session->m_shutdown_time, ATOMIC_CONSUME));
	if(shutdown_time < now){
		std::size_t send_buffer_size;
		{
			const Poseidon::Mutex::UniqueLock lock(session->m_send_mutex);
			send_buffer_size = session->m_send_buffer.size();
		}
		if(send_buffer_size == 0){
			LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_DEBUG,
				"Connection closed due to inactivity: remote = ", session->get_remote_info());
			session->set_timed_out();
			session->force_shutdown();
			return;
		}
		LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_DEBUG,
			"Shutdown pending: remote = ", session->get_remote_info(), ", send_buffer_size = ", send_buffer_size);
	}

	const AUTO(last_use_time, atomic_load(session->m_last_use_time, ATOMIC_CONSUME));
	const AUTO(tcp_response_timeout, MainConfig::get<boost::uint64_t>("tcp_response_timeout", 30000));
	if(saturated_add(last_use_time, tcp_response_timeout) < now){
		LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_DEBUG,
			"The connection seems dead: remote = ", session->get_remote_info());
		session->force_shutdown();
		return;
	}
}

TcpSessionBase::TcpSessionBase(UniqueFile socket)
	: SocketBase(STD_MOVE(socket)), SessionBase()
	, m_connected_notified(false), m_read_hup_notified(false)
	, m_shutdown_time((boost::uint64_t)-1), m_last_use_time((boost::uint64_t)-1)
{
}
TcpSessionBase::~TcpSessionBase(){
}

void TcpSessionBase::init_ssl(Move<boost::scoped_ptr<SslFilterBase> > ssl_filter){
	swap(m_ssl_filter, ssl_filter);
}
void TcpSessionBase::create_shutdown_timer(){
	PROFILE_ME;

	const Mutex::UniqueLock lock(m_shutdown_mutex);
	if(m_shutdown_timer){
		return;
	}
	m_shutdown_timer = TimerDaemon::register_low_level_timer(1000, 5000,
		boost::bind(&shutdown_timer_proc, virtual_weak_from_this<TcpSessionBase>(), _2));
}

int TcpSessionBase::poll_read_and_process(bool readable){
	PROFILE_ME;

	(void)readable;

	std::vector<unsigned char> temp;

	Poseidon::StreamBuffer data;
	try {
		temp.resize(4096);
		::ssize_t result;
		if(m_ssl_filter){
			result = m_ssl_filter->recv(temp.data(), temp.size());
		} else {
			result = ::recv(get_fd(), temp.data(), temp.size(), MSG_NOSIGNAL | MSG_DONTWAIT);
		}
		if(result < 0){
			return errno;
		}
		temp.resize(static_cast<std::size_t>(result));
		data.put(temp.data(), temp.size());
		LOG_POSEIDON_TRACE("Read ", result, " byte(s) from ", get_remote_info());

		const AUTO(now, get_fast_mono_clock());
		atomic_store(m_last_use_time, now, ATOMIC_RELEASE);
		create_shutdown_timer();

		if(data.empty() && !m_read_hup_notified){
			LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_DEBUG,
				"TCP connection read hung up: local = ", get_local_info(), ", remote = ", get_remote_info());
			shutdown_read();
			on_read_hup();
			m_read_hup_notified = true;
		}
		if(data.empty()){
			return EWOULDBLOCK;
		}
		on_receive(STD_MOVE(data));
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

	std::vector<unsigned char> temp;

	Poseidon::StreamBuffer data;
	try {
		if(writeable && !m_connected_notified){
			LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_DEBUG,
				"TCP connection established: local = ", get_local_info(), ", remote = ", get_remote_info());
			on_connect();
			m_connected_notified = true;
		}

		temp.resize(4096);
		Poseidon::Mutex::UniqueLock lock(m_send_mutex);
		const std::size_t avail = m_send_buffer.peek(temp.data(), temp.size());
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
		temp.resize(avail);

		::ssize_t result;
		if(m_ssl_filter){
			result = m_ssl_filter->send(temp.data(), temp.size());
		} else {
			result = ::send(get_fd(), temp.data(), temp.size(), MSG_NOSIGNAL | MSG_DONTWAIT);
		}
		if(result < 0){
			return errno;
		}
		LOG_POSEIDON_TRACE("Wrote ", result, " byte(s) to ", get_remote_info());

		const AUTO(now, get_fast_mono_clock());
		atomic_store(m_last_use_time, now, ATOMIC_RELEASE);
		create_shutdown_timer();

		lock.lock();
		m_send_buffer.discard(static_cast<std::size_t>(result));
		swap(write_lock, lock);
		if(m_send_buffer.empty()){
			return EWOULDBLOCK;
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
	return 0;
}

bool TcpSessionBase::is_throttled() const {
	const Mutex::UniqueLock lock(m_send_mutex);
	if(m_send_buffer.size() >= 65536){
		return true;
	}
	return SocketBase::is_throttled();
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

	const AUTO(now, get_fast_mono_clock());
	atomic_store(m_shutdown_time, saturated_add(now, timeout), ATOMIC_RELEASE);
	create_shutdown_timer();
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
