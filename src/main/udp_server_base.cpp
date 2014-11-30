// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014, LH_Mouse. All wrongs reserved.

#include "precompiled.hpp"
#include "udp_server_base.hpp"
#include "ip_port.hpp"
#include "sock_addr.hpp"
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include "exception.hpp"
#include "log.hpp"
using namespace Poseidon;

namespace {

ScopedFile createUdpSocket(const IpPort &addr){
    SockAddr sa = getSockAddrFromIpPort(addr);
    ScopedFile udp(::socket(sa.getFamily(), SOCK_DGRAM, IPPROTO_UDP));
    if(!udp){
        DEBUG_THROW(SystemError);
    }
    const int TRUE_VALUE = true;
    if(::setsockopt(udp.get(), SOL_SOCKET, SO_REUSEADDR, &TRUE_VALUE, sizeof(TRUE_VALUE)) != 0){
        DEBUG_THROW(SystemError);
    }
    if(::bind(udp.get(), static_cast<const ::sockaddr *>(sa.getData()), sa.getSize())){
        DEBUG_THROW(SystemError);
    }
    return udp;
}

}

UdpServerBase::UdpServerBase(const IpPort &bindAddr)
	: SocketServerBase(createUdpSocket(bindAddr))
{
	LOG_POSEIDON_INFO("Created UDP server on ", getLocalInfo());
}
UdpServerBase::~UdpServerBase(){
	LOG_POSEIDON_INFO("Destroyed UDP server on ", getLocalInfo());
}

bool UdpServerBase::poll() const {
	return false;
}
