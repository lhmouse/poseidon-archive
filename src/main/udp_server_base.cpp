#include "precompiled.hpp"
#include "udp_server_base.hpp"
#define POSEIDON_SOCK_ADDR_
#include "sock_addr.hpp"
using namespace Poseidon;

namespace {

ScopedFile createUdpSocket(const IpPort &addr){
	unsigned salen;
    SockAddr sa = getSockAddrFromIpPort(salen, addr);

    ScopedFile udp(::socket(sa.sa.sa_family, SOCK_DGRAM, IPPROTO_UDP));
    if(!udp){
        DEBUG_THROW(SystemError);
    }
    const int TRUE_VALUE = true;
    if(::setsockopt(udp.get(), SOL_SOCKET, SO_REUSEADDR, &TRUE_VALUE, sizeof(TRUE_VALUE)) != 0){
        DEBUG_THROW(SystemError);
    }
    if(::bind(udp.get(), &sa.sa, salen)){
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
