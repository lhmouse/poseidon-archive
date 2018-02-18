// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2018, LH_Mouse. All wrongs reserved.

#include "precompiled.hpp"
#include "udp_server_base.hpp"
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netinet/udp.h>
#include "system_exception.hpp"

namespace Poseidon {

namespace {
	UniqueFile create_udp_socket(const SockAddr &addr){
		UniqueFile udp;
		DEBUG_THROW_UNLESS(udp.reset(::socket(addr.get_family(), SOCK_DGRAM | SOCK_NONBLOCK, IPPROTO_UDP)), SystemException);
		static CONSTEXPR const int s_true_value = true;
		DEBUG_THROW_UNLESS(::setsockopt(udp.get(), SOL_SOCKET, SO_REUSEADDR, &s_true_value, sizeof(s_true_value)) == 0, SystemException);
		DEBUG_THROW_UNLESS(::bind(udp.get(), static_cast<const ::sockaddr *>(addr.data()), static_cast<unsigned>(addr.size())) == 0, SystemException);
		return udp;
	}
}

UdpServerBase::UdpServerBase(const SockAddr &addr)
	: UdpSessionBase(create_udp_socket(addr))
{
	LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_INFO, "Created UDP server on ", get_local_info());
}
UdpServerBase::~UdpServerBase(){
	LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_INFO, "Destroyed UDP server on ", get_local_info());
}

}
