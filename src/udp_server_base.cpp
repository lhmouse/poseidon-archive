// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2017, LH_Mouse. All wrongs reserved.

#include "precompiled.hpp"
#include "udp_server_base.hpp"
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <openssl/ssl.h>
#include "singletons/epoll_daemon.hpp"
#include "log.hpp"
#include "system_exception.hpp"
#include "profiler.hpp"

namespace Poseidon {

namespace {
#ifdef POSEIDON_CXX11
	UniqueFile
#else
	Move<UniqueFile>
#endif
		create_udp_socket(const SockAddr &addr)
	{
#ifdef POSEIDON_CXX11
		UniqueFile udp;
#else
		static __thread UniqueFile udp;
#endif
		if(!udp.reset(::socket(addr.get_family(), SOCK_DGRAM, IPPROTO_UDP))){
			DEBUG_THROW(SystemException);
		}
		static CONSTEXPR const int TRUE_VALUE = true;
		if(::setsockopt(udp.get(), SOL_SOCKET, SO_REUSEADDR, &TRUE_VALUE, sizeof(TRUE_VALUE)) != 0){
			DEBUG_THROW(SystemException);
		}
		if(::bind(udp.get(), static_cast<const ::sockaddr *>(addr.data()), addr.size()) != 0){
			DEBUG_THROW(SystemException);
		}
		return STD_MOVE(udp);
	}
}

UdpServerBase::UdpServerBase(const SockAddr &addr)
	: SocketBase(create_udp_socket(addr))
{
	LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_INFO,
		"Created UDP server on ", get_local_info());
}
UdpServerBase::~UdpServerBase(){
	LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_INFO,
		"Destroyed UDP server on ", get_local_info());
}

int UdpServerBase::poll_read_and_process(unsigned char *hint_buffer, std::size_t hint_capacity, bool readable){
	PROFILE_ME;

	(void)readable;

	for(unsigned i = 0; i < 256; ++i){
		SockAddr sock_addr;
		StreamBuffer data;
		try {
			::sockaddr_storage sa;
			::socklen_t sa_len = sizeof(sa);
			::ssize_t result = ::recvfrom(get_fd(), hint_buffer, hint_capacity, MSG_NOSIGNAL | MSG_DONTWAIT, static_cast< ::sockaddr *>(static_cast<void *>(&sa)), &sa_len);
			if(result < 0){
				return errno;
			}
			sock_addr = SockAddr(&sa, sa_len);
			data.put(hint_buffer, static_cast<std::size_t>(result));
			LOG_POSEIDON_TRACE("Read ", result, " byte(s) from ", IpPort(sock_addr));
		} catch(std::exception &e){
			LOG_POSEIDON_ERROR("std::exception thrown: what = ", e.what());
			return EINTR;
		}
		try {
			on_receive(sock_addr, STD_MOVE(data));
		} catch(std::exception &e){
			LOG_POSEIDON_ERROR("std::exception thrown: what = ", e.what());
			continue;
		} catch(...){
			LOG_POSEIDON_ERROR("Unknown exception thrown.");
			continue;
		}
	}
	return 0;
}
int UdpServerBase::poll_write(Mutex::UniqueLock &write_lock, unsigned char *hint_buffer, std::size_t hint_capacity, bool writeable){
	PROFILE_ME;

	(void)write_lock;
	(void)writeable;

	for(unsigned i = 0; i < 256; ++i){
		SockAddr sock_addr;
		StreamBuffer data;
		try {
			const Mutex::UniqueLock lock(m_send_mutex);
			if(m_send_queue.empty()){
				return EWOULDBLOCK;
			}
			sock_addr = m_send_queue.front().first;
			data.swap(m_send_queue.front().second);
			m_send_queue.pop_front();
		} catch(std::exception &e){
			LOG_POSEIDON_ERROR("std::exception thrown: what = ", e.what());
			return EINTR;
		}
		try {
			::sockaddr_storage sa;
			DEBUG_THROW_ASSERT(sock_addr.size() <= sizeof(sa));
			::socklen_t sa_len = sock_addr.size();
			const std::size_t avail = data.peek(hint_buffer, hint_capacity);
			if(avail < data.size()){
				LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_DEBUG, "UDP packet is too large: size = ", data.size());
		_too_large:
				on_message_too_large(sock_addr, STD_MOVE(data));
				continue;
			}
			::ssize_t result = ::sendto(get_fd(), hint_buffer, avail, MSG_NOSIGNAL | MSG_DONTWAIT, static_cast< ::sockaddr *>(static_cast<void *>(&sa)), sa_len);
			if(result < 0){
				if(errno == EMSGSIZE){
					LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_DEBUG, "UDP packet is too large: size = ", data.size());
					goto _too_large;
				}
				continue;
			}
			LOG_POSEIDON_TRACE("Wrote ", result, " byte(s) to ", IpPort(sock_addr));
		} catch(std::exception &e){
			LOG_POSEIDON_ERROR("std::exception thrown: what = ", e.what());
			continue;
		}
	}
	return 0;
}

void UdpServerBase::on_message_too_large(const SockAddr &sock_addr, StreamBuffer data) const {
	(void)sock_addr;
	(void)data;
}

bool UdpServerBase::send(const SockAddr &sock_addr, StreamBuffer buffer) const {
	PROFILE_ME;

	if(has_been_shutdown_write()){
		LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_DEBUG,
			"UDP socket has been shut down for writing: local = ", get_local_info(), ", remote = ", IpPort(sock_addr));
		return false;
	}

	const Mutex::UniqueLock lock(m_send_mutex);
	m_send_queue.emplace_back(sock_addr, STD_MOVE(buffer));
	EpollDaemon::mark_socket_writeable(this);
	return true;
}

}
