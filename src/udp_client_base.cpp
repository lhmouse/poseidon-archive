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
		DEBUG_THROW_UNLESS(udp.reset(::socket(addr.get_family(), SOCK_DGRAM, IPPROTO_UDP)), SystemException);
		return STD_MOVE(udp);
	}
}

UdpClientBase::UdpClientBase(const SockAddr &addr)
	: UdpSessionBase(create_udp_socket(addr))
{ }
UdpClientBase::~UdpClientBase(){ }

}
