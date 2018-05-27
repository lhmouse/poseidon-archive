// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2018, LH_Mouse. All wrongs reserved.

#include "precompiled.hpp"
#include "udp_client_base.hpp"
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netinet/udp.h>
#include "system_exception.hpp"

namespace Poseidon {

namespace {
	Unique_file create_udp_socket(const Sock_addr &addr){
		Unique_file udp;
		POSEIDON_THROW_UNLESS(udp.reset(::socket(addr.get_family(), SOCK_DGRAM | SOCK_NONBLOCK, IPPROTO_UDP)), System_exception);
		return udp;
	}
}

Udp_client_base::Udp_client_base(const Sock_addr &addr)
	: Udp_session_base(create_udp_socket(addr))
{
	//
}
Udp_client_base::~Udp_client_base(){
	//
}

}
