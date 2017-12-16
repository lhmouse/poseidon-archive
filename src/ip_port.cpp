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
	std::strcpy(m_ip, "???");
	m_port = 0;
}
IpPort::IpPort(const char *ip_str, unsigned port_num){
	const AUTO(ip_size, std::strlen(ip_str) + 1);
	DEBUG_THROW_UNLESS(ip_size <= sizeof(m_ip), Exception, sslit("IP address string is too long"));
	std::memcpy(m_ip, ip_str, ip_size);
	DEBUG_THROW_UNLESS(port_num <= 65535, Exception, sslit("Port number is too large"));
	m_port = port_num;
}
IpPort::IpPort(const SockAddr &sock_addr){
	const int family = sock_addr.get_family();
	if(family == AF_INET){
		const AUTO_REF(sin, *static_cast<const ::sockaddr_in *>(sock_addr.data()));
		BOOST_STATIC_ASSERT(sizeof(m_ip) >= INET_ADDRSTRLEN);
		DEBUG_THROW_UNLESS(::inet_ntop(AF_INET, &(sin.sin_addr), m_ip, sizeof(m_ip)), SystemException);
		m_port = load_be(sin.sin_port);
	} else if(family == AF_INET6){
		const AUTO_REF(sin6, *static_cast<const ::sockaddr_in6 *>(sock_addr.data()));
		BOOST_STATIC_ASSERT(sizeof(m_ip) >= INET6_ADDRSTRLEN);
		DEBUG_THROW_UNLESS(::inet_ntop(AF_INET6, &(sin6.sin6_addr), m_ip, sizeof(m_ip)), SystemException);
		m_port = load_be(sin6.sin6_port);
	} else {
		LOG_POSEIDON_WARNING("Unknown IP protocol ", family);
		DEBUG_THROW(Exception, sslit("Unknown IP protocol"));
	}
}
IpPort::IpPort(const IpPort &rhs) NOEXCEPT {
	std::strcpy(m_ip, rhs.m_ip);
	m_port = rhs.m_port;
}
IpPort &IpPort::operator=(const IpPort &rhs) NOEXCEPT {
	std::strcpy(m_ip, rhs.m_ip);
	m_port = rhs.m_port;
	return *this;
}

namespace {
	const IpPort g_unknown_ip_port("<unknown>", 0);
	const IpPort g_listening_ip_port("<listening>", 0);
}

const IpPort &unknown_ip_port() NOEXCEPT {
	return g_unknown_ip_port;
}
const IpPort &listening_ip_port() NOEXCEPT {
	return g_listening_ip_port;
}

bool operator<(const IpPort &lhs, const IpPort &rhs) NOEXCEPT {
	int cmp = std::strcmp(lhs.ip(), rhs.ip());
	if(cmp != 0){
		return cmp < 0;
	}
	return lhs.port() < rhs.port();
}

std::ostream &operator<<(std::ostream &os, const IpPort &rhs){
	return os <<rhs.ip() <<':' <<rhs.port();
}

}
