// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2018, LH_Mouse. All wrongs reserved.

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

namespace {
	inline ::sockaddr_in &as_sin(const void *data) NOEXCEPT {
		return *static_cast< ::sockaddr_in *>(const_cast<void *>(data));
	}
	inline ::sockaddr_in6 &as_sin6(const void *data) NOEXCEPT {
		return *static_cast< ::sockaddr_in6 *>(const_cast<void *>(data));
	}
}

Ip_port::Ip_port(){
	std::strcpy(m_ip, "???");
	m_port = 0;
}
Ip_port::Ip_port(const char *ip_str, boost::uint16_t port_num){
	const AUTO(ip_size, std::strlen(ip_str) + 1);
	DEBUG_THROW_UNLESS(ip_size <= sizeof(m_ip), Exception, sslit("IP address string is too long"));
	std::memcpy(m_ip, ip_str, ip_size);
	m_port = port_num;
}
Ip_port::Ip_port(const Sock_addr &sock_addr){
	const int family = sock_addr.get_family();
	switch(family){
	case AF_INET: {
		::sockaddr_in &sin = as_sin(sock_addr.data());
		DEBUG_THROW_UNLESS(sock_addr.size() >= sizeof(sin), Exception, sslit("Invalid IPv4 Sock_addr"));
		BOOST_STATIC_ASSERT(sizeof(m_ip) >= INET_ADDRSTRLEN);
		DEBUG_THROW_UNLESS(::inet_ntop(AF_INET, &(sin.sin_addr), m_ip, sizeof(m_ip)), System_exception);
		m_port = load_be(sin.sin_port);
		break; }
	case AF_INET6: {
		::sockaddr_in6 &sin6 = as_sin6(sock_addr.data());
		DEBUG_THROW_UNLESS(sock_addr.size() >= sizeof(sin6), Exception, sslit("Invalid IPv6 Sock_addr"));
		BOOST_STATIC_ASSERT(sizeof(m_ip) >= INET6_ADDRSTRLEN);
		DEBUG_THROW_UNLESS(::inet_ntop(AF_INET6, &(sin6.sin6_addr), m_ip, sizeof(m_ip)), System_exception);
		m_port = load_be(sin6.sin6_port);
		break; }
	default:
		LOG_POSEIDON_ERROR("Unknown IP protocol: family = ", family);
		DEBUG_THROW(Exception, sslit("Unknown IP protocol"));
	}
}
Ip_port::Ip_port(const Ip_port &rhs) NOEXCEPT {
	std::strcpy(m_ip, rhs.m_ip);
	m_port = rhs.m_port;
}
Ip_port &Ip_port::operator=(const Ip_port &rhs) NOEXCEPT {
	std::strcpy(m_ip, rhs.m_ip);
	m_port = rhs.m_port;
	return *this;
}

namespace {
	const Ip_port g_unknown_ip_port("<unknown>", 0);
	const Ip_port g_listening_ip_port("<listening>", 0);
}

const Ip_port &unknown_ip_port() NOEXCEPT {
	return g_unknown_ip_port;
}
const Ip_port &listening_ip_port() NOEXCEPT {
	return g_listening_ip_port;
}

bool operator<(const Ip_port &lhs, const Ip_port &rhs) NOEXCEPT {
	const int cmp = std::strcmp(lhs.ip(), rhs.ip());
	if(cmp != 0){
		return cmp < 0;
	}
	return lhs.port() < rhs.port();
}

std::ostream &operator<<(std::ostream &os, const Ip_port &rhs){
	return os <<rhs.ip() <<':' <<rhs.port();
}

}
