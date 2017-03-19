// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2017, LH_Mouse. All wrongs reserved.

#include "precompiled.hpp"
#include "ip_port.hpp"
#include <ostream>
#include <arpa/inet.h>
#include <sys/types.h>
#include <sys/socket.h>
#include "sock_addr.hpp"
#include "endian.hpp"
#include "system_exception.hpp"
#include "log.hpp"

namespace Poseidon {

IpPort::IpPort(){
	std::memcpy(m_ip, "<unknown>", 10);
	m_port = 0;
}
IpPort::IpPort(const char *ip_p, unsigned port_p){
	const AUTO(ip_size, std::strlen(ip_p) + 1);
	if(ip_size > sizeof(m_ip)){
		LOG_POSEIDON_ERROR("IP address is too long: ip_p = ", ip_p);
		DEBUG_THROW(Exception, sslit("IP address is too long"));
	}
	std::memcpy(m_ip, ip_p, ip_size);
	m_port = port_p;
}

namespace {
	const IpPort g_unknown_ip_port;
}

const IpPort &unknown_ip_port() NOEXCEPT {
	return g_unknown_ip_port;
}

std::ostream &operator<<(std::ostream &os, const IpPort &rhs){
	return os <<rhs.ip() <<':' <<rhs.port();
}

IpPort get_ip_port_from_sock_addr(const SockAddr &sa){
	const int family = sa.get_family();
	if(family == AF_INET){
		if(sa.size() < sizeof(::sockaddr_in)){
			LOG_POSEIDON_WARNING("Invalid IPv4 SockAddr: size = ", sa.size());
			DEBUG_THROW(Exception, sslit("Invalid IPv4 SockAddr"));
		}
		char ip[INET_ADDRSTRLEN];
		const char *const str = ::inet_ntop(AF_INET, &(static_cast<const ::sockaddr_in *>(sa.data())->sin_addr), ip, sizeof(ip));
		if(!str){
			DEBUG_THROW(SystemException);
		}
		return IpPort(str, load_be(static_cast<const ::sockaddr_in *>(sa.data())->sin_port));
	} else if(family == AF_INET6){
		if(sa.size() < sizeof(::sockaddr_in6)){
			LOG_POSEIDON_WARNING("Invalid IPv6 SockAddr: size = ", sa.size());
			DEBUG_THROW(Exception, sslit("Invalid IPv6 SockAddr"));
		}
		char ip[INET6_ADDRSTRLEN];
		const char *const str = ::inet_ntop(AF_INET6, &(static_cast<const ::sockaddr_in6 *>(sa.data())->sin6_addr), ip, sizeof(ip));
		if(!str){
			DEBUG_THROW(SystemException);
		}
		return IpPort(str, load_be(static_cast<const ::sockaddr_in6 *>(sa.data())->sin6_port));
	} else {
		LOG_POSEIDON_WARNING("Unknown IP protocol ", family);
		DEBUG_THROW(Exception, sslit("Unknown IP protocol"));
	}
}

}
