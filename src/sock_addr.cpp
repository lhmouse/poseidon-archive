// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2017, LH_Mouse. All wrongs reserved.

#include "precompiled.hpp"
#include "sock_addr.hpp"
#include <arpa/inet.h>
#include <sys/types.h>
#include <sys/socket.h>
#include "ip_port.hpp"
#include "endian.hpp"
#include "system_exception.hpp"
#include "log.hpp"

namespace Poseidon {

namespace {
	bool is_ipv4_private(const unsigned char *ip){
		if(ip[0] == 0){ // 0.0.0.0/8: 当前网络地址
			return true;
		} else if(ip[0] == 10){ // 10.0.0.0/8: A 类私有地址
			return true;
		} else if(ip[0] == 127){ // 127.0.0.0/8: 回环地址
			return true;
		} else if((ip[0] == 172) && ((ip[1] & 0xF0) == 16)){ // 172.16.0.0/12: B 类私有地址
			return true;
		} else if((ip[0] == 169) && (ip[1] == 254)){ // 169.254.0.0/16: 链路本地地址
			return true;
		} else if((ip[0] == 192) && (ip[1] == 168)){ // 192.168.0.0/16: C 类私有地址
			return true;
		} else if(ip[0] >= 224){ // D 类、E 类地址和广播地址
			return true;
		} else {
			return false;
		}
	}
	bool are_zeroes(const void *p, unsigned len){
		static CONSTEXPR const unsigned char ZEROES[16] = { };
		assert(len <= sizeof(ZEROES));
		return std::memcmp(p, ZEROES, len) == 0;
	}
}

SockAddr::SockAddr(){
	static CONSTEXPR const ::sa_family_t NULL_FAMILY = AF_UNSPEC;
	std::memcpy(m_data, &NULL_FAMILY, sizeof(NULL_FAMILY));
	m_size = 0;
}
SockAddr::SockAddr(const void *addr_data, std::size_t addr_size){
	DEBUG_THROW_UNLESS(addr_size <= sizeof(m_data), Exception, sslit("SockAddr size too large"));
	std::memcpy(m_data, addr_data, addr_size);
	m_size = addr_size;
}
SockAddr::SockAddr(const IpPort &ip_port){
	if(::inet_pton(AF_INET, ip_port.ip(), &(static_cast< ::sockaddr_in *>(static_cast<void *>(m_data))->sin_addr)) == 1){
		BOOST_STATIC_ASSERT(sizeof(m_data) >= sizeof(::sockaddr_in));
		AUTO_REF(sin, *static_cast< ::sockaddr_in *>(static_cast<void *>(m_data)));
		sin.sin_family = AF_INET;
		store_be(sin.sin_port, ip_port.port());
		m_size = sizeof(sin);
	} else if(::inet_pton(AF_INET6, ip_port.ip(), &(static_cast< ::sockaddr_in6 *>(static_cast<void *>(m_data))->sin6_addr)) == 1){
		BOOST_STATIC_ASSERT(sizeof(m_data) >= sizeof(::sockaddr_in6));
		AUTO_REF(sin6, *static_cast< ::sockaddr_in6 *>(static_cast<void *>(m_data)));
		sin6.sin6_family = AF_INET6;
		store_be(sin6.sin6_port, ip_port.port());
		m_size = sizeof(sin6);
	} else {
		LOG_POSEIDON_ERROR("Unknown IP address format: ip = ", ip_port.ip());
		DEBUG_THROW(Exception, sslit("Unknown IP address format"));
	}
}
SockAddr::SockAddr(const SockAddr &rhs) NOEXCEPT {
	std::memcpy(m_data, rhs.m_data, rhs.m_size);
	m_size = rhs.m_size;
}
SockAddr &SockAddr::operator=(const SockAddr &rhs) NOEXCEPT {
	std::memcpy(m_data, rhs.m_data, rhs.m_size);
	m_size = rhs.m_size;
	return *this;
}

int SockAddr::get_family() const {
	::sa_family_t sa_family;
	if(m_size < sizeof(sa_family)){
		LOG_POSEIDON_ERROR("Invalid SockAddr: size = ", m_size);
		DEBUG_THROW(Exception, sslit("Invalid SockAddr"));
	}
	std::memcpy(&sa_family, m_data, sizeof(sa_family));
	return sa_family;
}
bool SockAddr::is_ipv6() const {
	const int family = get_family();
	if(family == AF_INET){
		return false;
	} else if(family == AF_INET6){
		return true;
	} else {
		LOG_POSEIDON_WARNING("Unknown IP protocol ", family);
		DEBUG_THROW(Exception, sslit("Unknown IP protocol"));
	}
}
bool SockAddr::is_private() const {
	const int family = get_family();
	if(family == AF_INET){
		const AUTO_REF(sin, *static_cast<const ::sockaddr_in *>(static_cast<const void *>(m_data)));
		const AUTO(ip, reinterpret_cast<const unsigned char *>(&(sin.sin_addr)));
		return is_ipv4_private(ip);
	} else if(family == AF_INET6){
		const AUTO_REF(sin6, *static_cast<const ::sockaddr_in6 *>(static_cast<const void *>(m_data)));
		const AUTO(ip, reinterpret_cast<const unsigned char *>(&(sin6.sin6_addr)));
		if(are_zeroes(ip, 15)){
			if(ip[15] == 0){ // ::/128: 未指定的地址
				return true;
			} else if(ip[15] == 1){ // ::1/128: 回环地址
				return true;
			} else {
				return false;
			}
		} else if(are_zeroes(ip, 10) && (ip[10] == 0xFF) && (ip[11] == 0xFF)){ // IPv4 翻译地址
			return is_ipv4_private(ip + 12);
		} else if((ip[0] == 0x01) && are_zeroes(ip + 1, 7)){ // 100::/64 黑洞地址
			return true;
		} else if(ip[0] >= 0xFC){ // 私有地址、链路本地地址和广播地址
			return true;
		} else {
			return false;
		}
	} else {
		LOG_POSEIDON_WARNING("Unknown IP protocol: family = ", family);
		DEBUG_THROW(Exception, sslit("Unknown IP protocol"));
	}
}

}
