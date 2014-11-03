#include "precompiled.hpp"
#include "socket_server_base.hpp"
#define POSEIDON_SOCK_ADDR_
#include "sock_addr.hpp"
#include <fcntl.h>
using namespace Poseidon;

namespace {

IpPort getAddrPortFromFd(int fd){
	SockAddr sa;
	::socklen_t salen = sizeof(sa);
	if(::getsockname(fd, &sa.sa, &salen) != 0){
		const int code = errno;
		LOG_POSEIDON_ERROR("Failed to get local socket addr.");
		DEBUG_THROW(SystemError, code);
	}
	return getIpPortFromSockAddr(sa);
}

}

SocketServerBase::SocketServerBase(ScopedFile socket)
	: m_socket(STD_MOVE(socket)), m_localInfo(getAddrPortFromFd(m_socket.get()))
{
	const int flags = ::fcntl(m_socket.get(), F_GETFL);
	if(flags == -1){
		const int code = errno;
		LOG_POSEIDON_ERROR("Could not get fcntl flags on socket.");
		DEBUG_THROW(SystemError, code);
	}
	if(::fcntl(m_socket.get(), F_SETFL, flags | O_NONBLOCK) != 0){
		const int code = errno;
		LOG_POSEIDON_ERROR("Could not set fcntl flags on socket.");
		DEBUG_THROW(SystemError, code);
	}

	LOG_POSEIDON_INFO("Created socket server, local = ", m_localInfo);
}
SocketServerBase::~SocketServerBase(){
	LOG_POSEIDON_INFO("Destroyed socket server, local = ", m_localInfo);
}
