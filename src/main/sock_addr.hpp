#ifndef POSEIDON_SOCK_ADDR_
#	error Please do not #include "sock_addr.hpp".
#endif

#ifndef POSEIDON_SOCK_ADDR_HPP_
#define POSEIDON_SOCK_ADDR_HPP_

#include <arpa/inet.h>
#include <sys/types.h>
#include <sys/socket.h>
#include "ip_port.hpp"
#include "endian.hpp"
#include "exception.hpp"
#include "log.hpp"

namespace Poseidon {

union SockAddr {
	::sockaddr sa;
	::sockaddr_in sin;
	::sockaddr_in6 sin6;
};

inline IpPort getIpPortFromSockAddr(const SockAddr &sa){
	char ip[64];
	unsigned port;
	const char *ret;
	if(sa.sa.sa_family == AF_INET){
		ret = ::inet_ntop(AF_INET, &sa.sin.sin_addr, ip, sizeof(ip));
		port = loadBe(sa.sin.sin_port);
	} else if(sa.sa.sa_family == AF_INET6){
		ret = ::inet_ntop(AF_INET6, &sa.sin6.sin6_addr, ip, sizeof(ip));
		port = loadBe(sa.sin6.sin6_port);
	} else {
		LOG_POSEIDON_WARN("Unknown IP protocol ", sa.sa.sa_family);
		DEBUG_THROW(Exception, "Unknown IP protocol");
	}
	if(!ret){
		const int code = errno;
		LOG_POSEIDON_WARN("Failed to format IP address to string.");
		DEBUG_THROW(SystemError, code);
	}
	return IpPort(SharedNtmbs(ip, true), port);
}

inline SockAddr getSockAddrFromIpPort(unsigned &salen, const IpPort &addr){
	SockAddr sa;
	if(::inet_pton(AF_INET, addr.ip.get(), &sa.sin.sin_addr) == 1){
		sa.sin.sin_family = AF_INET;
		storeBe(sa.sin.sin_port, addr.port);
		salen = sizeof(::sockaddr_in);
	} else if(::inet_pton(AF_INET6, addr.ip.get(), &sa.sin6.sin6_addr) == 1){
		sa.sin6.sin6_family = AF_INET6;
		storeBe(sa.sin6.sin6_port, addr.port);
		salen = sizeof(::sockaddr_in6);
	} else {
		LOG_POSEIDON_ERROR("Unknown address format: ", addr.ip);
		DEBUG_THROW(Exception, "Unknown address format");
	}
	return sa;
}

}

#endif
