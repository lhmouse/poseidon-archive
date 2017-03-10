// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2017, LH_Mouse. All wrongs reserved.

#include "precompiled.hpp"
#include "udp_server_base.hpp"
#include "ip_port.hpp"
#include "sock_addr.hpp"
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include "system_exception.hpp"
#include "log.hpp"
#include "profiler.hpp"
#include "atomic.hpp"

namespace Poseidon {

namespace {
	enum {
		MAX_PUMP_COUNT          = 65536,
		IO_BUFFER_SIZE          = 4096,
	};

	UniqueFile create_udp_socket(const SockAddr &addr){
		UniqueFile udp(::socket(addr.get_family(), SOCK_DGRAM, IPPROTO_UDP));
		if(!udp){
			DEBUG_THROW(SystemException);
		}
		const int TRUE_VALUE = true;
		if(::setsockopt(udp.get(), SOL_SOCKET, SO_REUSEADDR, &TRUE_VALUE, sizeof(TRUE_VALUE)) != 0){
			DEBUG_THROW(SystemException);
		}
		if(::bind(udp.get(), static_cast<const ::sockaddr *>(addr.get_data()), addr.get_size())){
			DEBUG_THROW(SystemException);
		}
		return udp;
	}
}

UdpServerBase::UdpServerBase(const SockAddr &addr)
	: SocketServerBase(create_udp_socket(addr))
	, m_shutdown_read(false), m_shutdown_write(false)
{
	LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_INFO, "Created UDP server on ", get_local_info());
}
UdpServerBase::UdpServerBase(const IpPort &addr)
	: SocketServerBase(create_udp_socket(get_sock_addr_from_ip_port(addr)))
	, m_shutdown_read(false), m_shutdown_write(false)
{
	LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_INFO, "Created UDP server on ", get_local_info());
}
UdpServerBase::~UdpServerBase(){
	LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_INFO, "Destroyed UDP server on ", get_local_info_nothrow());
}

UdpServerBase::SyncIoResult UdpServerBase::sync_read_and_process(void *hint, unsigned long hint_size) const {
	PROFILE_ME;

	SyncIoResult ret;
	::sockaddr sa;
	::socklen_t sa_len = sizeof(sa);
	ret.bytes_transferred = ::recvfrom(get_fd(), hint, hint_size, MSG_NOSIGNAL | MSG_DONTWAIT, &sa, &sa_len);
	if(ret.bytes_transferred < 0){
		ret.err_code = errno;
	} else {
		ret.err_code = 0;
	}
	if(ret.bytes_transferred >= 0){
		const SockAddr sock_addr(&sa, sa_len);
		const AUTO(bytes_transferred, static_cast<std::size_t>(ret.bytes_transferred));
		LOG_POSEIDON_TRACE("Read ", bytes_transferred, " byte(s) from ", get_ip_port_from_sock_addr(sock_addr));

		on_receive(sock_addr, StreamBuffer(hint, bytes_transferred));
	}
	return ret;
}
UdpServerBase::SyncIoResult UdpServerBase::sync_write(void *hint, unsigned long hint_size) const {
	PROFILE_ME;

	SockAddr sock_addr;
	std::size_t bytes_avail = 0;
	{
		const Mutex::UniqueLock lock(m_send_mutex);
		if(!m_send_queue.empty()){
			const AUTO_REF(pair, m_send_queue.front());
			sock_addr = pair.first;
			bytes_avail = pair.second.peek(hint, hint_size);
		}
	}

	SyncIoResult ret;
	if(sock_addr.get_size() == 0){
		ret.bytes_transferred = -1;
		ret.err_code = EAGAIN;
	} else {
		::sockaddr sa;
		::socklen_t sa_len = std::min< ::socklen_t>(sizeof(sa), sock_addr.get_size());
		std::memcpy(&sa, sock_addr.get_data(), sa_len);
		ret.bytes_transferred = ::sendto(get_fd(), hint, bytes_avail, MSG_NOSIGNAL | MSG_DONTWAIT, &sa, sa_len);
		if(ret.bytes_transferred < 0){
			ret.err_code = errno;
		} else {
			ret.err_code = 0;
		}
	}
	if(ret.bytes_transferred >= 0){
		const AUTO(bytes_transferred, static_cast<std::size_t>(ret.bytes_transferred));
		LOG_POSEIDON_TRACE("Wrote ", bytes_transferred, " byte(s) to ", get_ip_port_from_sock_addr(sock_addr));

		if(bytes_transferred < bytes_avail){
			LOG_POSEIDON_WARNING("UDP packet truncated: bytes_transferred = ", bytes_transferred, ", bytes_avail = ", bytes_avail);
		}

		const Mutex::UniqueLock lock(m_send_mutex);
		m_send_queue.pop_front();
	}
	return ret;
}

bool UdpServerBase::has_been_shutdown_read() const NOEXCEPT {
	return atomic_load(m_shutdown_read, ATOMIC_CONSUME);
}
bool UdpServerBase::shutdown_read() NOEXCEPT {
	PROFILE_ME;

	bool ret = !atomic_load(m_shutdown_read, ATOMIC_CONSUME);
	if(ret){
		ret = !atomic_exchange(m_shutdown_read, true, ATOMIC_ACQ_REL);
		::shutdown(get_fd(), SHUT_RD);
	}
	return ret;
}
bool UdpServerBase::has_been_shutdown_write() const NOEXCEPT {
	return atomic_load(m_shutdown_write, ATOMIC_CONSUME);
}
bool UdpServerBase::shutdown_write() NOEXCEPT {
	PROFILE_ME;

	bool ret = !atomic_load(m_shutdown_write, ATOMIC_CONSUME);
	if(ret){
		ret = !atomic_exchange(m_shutdown_write, true, ATOMIC_ACQ_REL);
		::shutdown(get_fd(), SHUT_WR);
	}
	return ret;
}
void UdpServerBase::force_shutdown() NOEXCEPT {
	PROFILE_ME;

	atomic_store(m_shutdown_read, true, ATOMIC_RELEASE);
	atomic_store(m_shutdown_write, true, ATOMIC_RELEASE);

	::linger lng;
	lng.l_onoff = 1;
	lng.l_linger = 0;
	::setsockopt(get_fd(), SOL_SOCKET, SO_LINGER, &lng, sizeof(lng));

	::shutdown(get_fd(), SHUT_RDWR);
}

bool UdpServerBase::poll() const {
	PROFILE_ME;

	bool busy = false;

	for(std::size_t i = 0; i < MAX_PUMP_COUNT; ++i){
		try {
			unsigned char temp[IO_BUFFER_SIZE];
			const AUTO(result, sync_read_and_process(temp, sizeof(temp)));
			if(result.bytes_transferred < 0){
				if(result.err_code == EINTR){
					continue;
				}
				if(result.err_code == EAGAIN){
					break;
				}
				DEBUG_THROW(SystemException, result.err_code);
			}
			++busy;
		} catch(std::exception &e){
			LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_INFO,
				"std::exception thrown: what = ", e.what(), ", typeid = ", typeid(*this).name());
		} catch(...){
			LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_INFO,
				"Unknown exception thrown: typeid = ", typeid(*this).name());
		}
	}

	for(std::size_t i = 0; i < MAX_PUMP_COUNT; ++i){
		try {
			unsigned char temp[IO_BUFFER_SIZE];
			const AUTO(result, sync_write(temp, sizeof(temp)));
			if(result.bytes_transferred < 0){
				if(result.err_code == EINTR){
					continue;
				}
				if(result.err_code == EAGAIN){
					break;
				}
				DEBUG_THROW(SystemException, result.err_code);
			}
			++busy;
		} catch(std::exception &e){
			LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_INFO,
				"std::exception thrown: what = ", e.what(), ", typeid = ", typeid(*this).name());
		} catch(...){
			LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_INFO,
				"Unknown exception thrown: typeid = ", typeid(*this).name());
		}
	}

	return busy;
}

bool UdpServerBase::send(const SockAddr &sock_addr, StreamBuffer buffer) const {
	PROFILE_ME;

	if(atomic_load(m_shutdown_write, ATOMIC_CONSUME)){
		LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_DEBUG,
			"UDP socket has been shut down for writing: local = ", get_local_info_nothrow());
		return false;
	}

	const Mutex::UniqueLock lock(m_send_mutex);
	m_send_queue.emplace_back(sock_addr, STD_MOVE(buffer));
	return true;
}

}
