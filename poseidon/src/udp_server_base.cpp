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
	Unique_file create_udp_socket(const Sock_addr &addr){
		Unique_file udp;
		DEBUG_THROW_UNLESS(udp.reset(::socket(addr.get_family(), SOCK_DGRAM | SOCK_NONBLOCK, IPPROTO_UDP)), System_exception);
		static CONSTEXPR const int s_true_value = true;
		DEBUG_THROW_UNLESS(::setsockopt(udp.get(), SOL_SOCKET, SO_REUSEADDR, &s_true_value, sizeof(s_true_value)) == 0, System_exception);
		DEBUG_THROW_UNLESS(::bind(udp.get(), static_cast<const ::sockaddr *>(addr.data()), static_cast<unsigned>(addr.size())) == 0, System_exception);
		return udp;
	}
}

Udp_server_base::Udp_server_base(const Sock_addr &addr)
	: Udp_session_base(create_udp_socket(addr))
{
	LOG_POSEIDON(Logger::special_major | Logger::level_info, "Created UDP server on ", get_local_info());
}
Udp_server_base::~Udp_server_base(){
	LOG_POSEIDON(Logger::special_major | Logger::level_info, "Destroyed UDP server on ", get_local_info());
}

}
