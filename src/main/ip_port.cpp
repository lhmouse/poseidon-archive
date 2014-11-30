// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014, LH_Mouse. All wrongs reserved.

#include "precompiled.hpp"
#include "ip_port.hpp"
#include <ostream>
#include <sys/types.h>
#include <sys/socket.h>
#include "sock_addr.hpp"
#include "exception.hpp"
using namespace Poseidon;

namespace Poseidon {

std::ostream &operator<<(std::ostream &os, const IpPort &rhs){
	return os <<rhs.ip <<':' <<rhs.port;
}
std::wostream &operator<<(std::wostream &os, const IpPort &rhs){
	return os <<rhs.ip <<':' <<rhs.port;
}

IpPort getRemoteIpPortFromFd(int fd){
	::sockaddr sa;
	::socklen_t salen = sizeof(sa);
	if(::getpeername(fd, &sa, &salen) != 0){
		DEBUG_THROW(SystemError);
	}
	return getIpPortFromSockAddr(SockAddr(&sa, salen));
}
IpPort getLocalIpPortFromFd(int fd){
	::sockaddr sa;
	::socklen_t salen = sizeof(sa);
	if(::getsockname(fd, &sa, &salen) != 0){
		DEBUG_THROW(SystemError);
	}
	return getIpPortFromSockAddr(SockAddr(&sa, salen));
}

}
