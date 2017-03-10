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

namespace Poseidon {

namespace {
	enum {
		IO_BUFFER_SIZE          = 16384,
	};

	UniqueFile create_udp_socket(const IpPort &addr){
		SockAddr sa = get_sock_addr_from_ip_port(addr);
		UniqueFile udp(::socket(sa.get_family(), SOCK_DGRAM, IPPROTO_UDP));
		if(!udp){
			DEBUG_THROW(SystemException);
		}
		const int TRUE_VALUE = true;
		if(::setsockopt(udp.get(), SOL_SOCKET, SO_REUSEADDR, &TRUE_VALUE, sizeof(TRUE_VALUE)) != 0){
			DEBUG_THROW(SystemException);
		}
		if(::bind(udp.get(), static_cast<const ::sockaddr *>(sa.get_data()), sa.get_size())){
			DEBUG_THROW(SystemException);
		}
		return udp;
	}
}

UdpServerBase::UdpServerBase(const IpPort &bind_addr)
	: SocketServerBase(create_udp_socket(bind_addr))
{
	LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_INFO, "Created UDP server on ", get_local_info());
}
UdpServerBase::~UdpServerBase(){
	LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_INFO, "Destroyed UDP server on ", get_local_info_nothrow());
}

bool UdpServerBase::poll() const {
	unsigned char temp[IO_BUFFER_SIZE];
	::sockaddr src_addr;
	::socklen_t src_addr_len = sizeof(src_addr);
	AUTO(bytes_transferred, ::recvfrom(get_fd(), temp, sizeof(temp), MSG_NOSIGNAL | MSG_DONTWAIT, &src_addr, &src_addr_len));
	if(bytes_transferred < 0){
		return false;
	}

	const AUTO(sock_addr, SockAddr(&src_addr, src_addr_len));
	const AUTO(bytes, static_cast<std::size_t>(bytes_transferred));
	LOG_POSEIDON_TRACE("Read ", bytes, " byte(s) from ", get_ip_port_from_sock_addr(sock_addr));

	try {
		on_receive(sock_addr, StreamBuffer(temp, bytes));
	} catch(std::exception &e){
		LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_INFO,
			"std::exception thrown while reading socket: what = ", e.what(), ", typeid = ", typeid(*this).name());
	} catch(...){
		LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_INFO,
			"Unknown exception thrown while reading socket: typeid = ", typeid(*this).name());
	}
	return true;
}
/*
bool UdpServerBase::send(const SockAddr &sock_addr, StreamBuffer data) const {
	PROFILE_ME;

	
}
*/
}
