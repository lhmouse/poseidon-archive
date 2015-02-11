// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2015, LH_Mouse. All wrongs reserved.

#include "precompiled.hpp"
#include "sock_addr.hpp"
#include <arpa/inet.h>
#include <sys/types.h>
#include <sys/socket.h>
#include "ip_port.hpp"
#include "endian.hpp"
#include "exception.hpp"
#include "log.hpp"
using namespace Poseidon;

SockAddr::SockAddr(){
	m_size = 0;
}
SockAddr::SockAddr(const void *data, unsigned size){
	if(size > sizeof(m_data)){
		LOG_POSEIDON_ERROR("SockAddr size too large: ", size);
		DEBUG_THROW(Exception, SharedNts::observe("SockAddr size too large"));
	}
	m_size = size;
	std::memcpy(m_data, data, size);
}

int SockAddr::getFamily() const {
	const AUTO(p, reinterpret_cast<const ::sockaddr *>(m_data));
	if(m_size < sizeof(p->sa_family)){
		LOG_POSEIDON_ERROR("Invalid SockAddr: size = ", m_size);
		DEBUG_THROW(Exception, SharedNts::observe("Invalid SockAddr"));
	}
	return p->sa_family;
}

namespace Poseidon {

IpPort getIpPortFromSockAddr(const SockAddr &sa){
	const int family = sa.getFamily();
	if(family == AF_INET){
		if(sa.getSize() < sizeof(::sockaddr_in)){
			LOG_POSEIDON_WARNING("Invalid IPv4 SockAddr: size = ", sa.getSize());
			DEBUG_THROW(Exception, SharedNts::observe("Invalid IPv4 SockAddr"));
		}
		char ip[INET_ADDRSTRLEN];
		const char *const str = ::inet_ntop(AF_INET,
			&static_cast<const ::sockaddr_in *>(sa.getData())->sin_addr, ip, sizeof(ip));
		if(!str){
			DEBUG_THROW(SystemError);
		}
		return IpPort(SharedNts(str),
			loadBe(static_cast<const ::sockaddr_in *>(sa.getData())->sin_port));
	} else if(family == AF_INET6){
		if(sa.getSize() < sizeof(::sockaddr_in6)){
			LOG_POSEIDON_WARNING("Invalid IPv6 SockAddr: size = ", sa.getSize());
			DEBUG_THROW(Exception, SharedNts::observe("Invalid IPv6 SockAddr"));
		}
		char ip[INET6_ADDRSTRLEN];
		const char *const str = ::inet_ntop(AF_INET6,
			&static_cast<const ::sockaddr_in6 *>(sa.getData())->sin6_addr, ip, sizeof(ip));
		if(!str){
			DEBUG_THROW(SystemError);
		}
		return IpPort(SharedNts(str),
			loadBe(static_cast<const ::sockaddr_in6 *>(sa.getData())->sin6_port));
	}

	LOG_POSEIDON_WARNING("Unknown IP protocol ", family);
	DEBUG_THROW(Exception, SharedNts::observe("Unknown IP protocol"));
}
SockAddr getSockAddrFromIpPort(const IpPort &addr){
	::sockaddr_in sin;
	if(::inet_pton(AF_INET, addr.ip.get(), &sin.sin_addr) == 1){
		sin.sin_family = AF_INET;
		storeBe(sin.sin_port, addr.port);
		return SockAddr(&sin, sizeof(sin));
	}

	::sockaddr_in6 sin6;
	if(::inet_pton(AF_INET6, addr.ip.get(), &sin6.sin6_addr) == 1){
		sin6.sin6_family = AF_INET6;
		storeBe(sin6.sin6_port, addr.port);
		return SockAddr(&sin6, sizeof(sin6));
	}

	LOG_POSEIDON_ERROR("Unknown address format: ", addr.ip);
	DEBUG_THROW(Exception, SharedNts::observe("Unknown address format"));
}

}
