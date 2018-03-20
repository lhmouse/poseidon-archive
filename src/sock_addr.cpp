// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2018, LH_Mouse. All wrongs reserved.

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
	inline bool is_ipv4_private(const unsigned char *addr){
		if(addr[0] == 0){ // 0.0.0.0/8: 当前网络地址
			return true;
		} else if(addr[0] == 10){ // 10.0.0.0/8: A 类私有地址
			return true;
		} else if(addr[0] == 127){ // 127.0.0.0/8: 回环地址
			return true;
		} else if((addr[0] == 172) && ((addr[1] & 0xF0) == 16)){ // 172.16.0.0/12: B 类私有地址
			return true;
		} else if((addr[0] == 169) && (addr[1] == 254)){ // 169.254.0.0/16: 链路本地地址
			return true;
		} else if((addr[0] == 192) && (addr[1] == 168)){ // 192.168.0.0/16: C 类私有地址
			return true;
		} else if(addr[0] >= 224){ // D 类、E 类地址和广播地址
			return true;
		}
		return false;
	}
	inline bool is_ipv4_multicast(const unsigned char *addr){
		return (addr[0] & 0xF0) == 224;
	}
	inline bool are_zeroes(const unsigned char *data, std::size_t size){
#ifdef POSEIDON_CXX11
		return std::all_of(data, data + size, [](unsigned char byte){ return byte == 0; });
#else
		return std::count(data, data + size, static_cast<unsigned char>(0)) == size;
#endif
	}

	inline ::sockaddr &as_sa(const void *data) NOEXCEPT {
		return *static_cast< ::sockaddr *>(const_cast<void *>(data));
	}
	inline ::sockaddr_in &as_sin(const void *data) NOEXCEPT {
		return *static_cast< ::sockaddr_in *>(const_cast<void *>(data));
	}
	inline ::sockaddr_in6 &as_sin6(const void *data) NOEXCEPT {
		return *static_cast< ::sockaddr_in6 *>(const_cast<void *>(data));
	}
}

SockAddr::SockAddr(){
	as_sa(m_data).sa_family = AF_UNSPEC;
	m_size = 0;
}
SockAddr::SockAddr(const void *addr_data, std::size_t addr_size){
	DEBUG_THROW_UNLESS(addr_size <= sizeof(m_data), Exception, sslit("Too many bytes for SockAddr"));
	std::memcpy(m_data, addr_data, addr_size);
	m_size = addr_size;
}
SockAddr::SockAddr(const IpPort &ip_port){
	if(::inet_pton(AF_INET, ip_port.ip(), &(as_sin(m_data).sin_addr)) == 1){
		::sockaddr_in &sin = as_sin(m_data);
		BOOST_STATIC_ASSERT(sizeof(m_data) >= sizeof(sin));
		sin.sin_family = AF_INET;
		store_be(sin.sin_port, ip_port.port());
		m_size = sizeof(sin);
	} else if(::inet_pton(AF_INET6, ip_port.ip(), &(as_sin6(m_data).sin6_addr)) == 1){
		::sockaddr_in6 &sin6 = as_sin6(m_data);
		BOOST_STATIC_ASSERT(sizeof(m_data) >= sizeof(sin6));
		sin6.sin6_family = AF_INET6;
		store_be(sin6.sin6_port, ip_port.port());
		m_size = sizeof(sin6);
	} else {
		LOG_POSEIDON_ERROR("Unknown IP address format: ip = ", ip_port.ip());
		DEBUG_THROW(Exception, sslit("Unknown IP address format"));
	}
}
SockAddr::SockAddr(const SockAddr &rhs) NOEXCEPT {
	const std::size_t addr_size = rhs.m_size;
	std::memcpy(m_data, rhs.m_data, addr_size);
	m_size = addr_size;
}
SockAddr &SockAddr::operator=(const SockAddr &rhs) NOEXCEPT {
	const std::size_t addr_size = rhs.m_size;
	std::memcpy(m_data, rhs.m_data, addr_size);
	m_size = addr_size;
	return *this;
}

int SockAddr::get_family() const {
	::sockaddr &sa = as_sa(m_data);
	DEBUG_THROW_UNLESS(m_size >= sizeof(sa.sa_family), Exception, sslit("Empty SockAddr"));
	return sa.sa_family;
}
bool SockAddr::is_ipv6() const {
	const int family = get_family();
	switch(family){
	case AF_INET:
		return false;
	case AF_INET6:
		return true;
	default:
		LOG_POSEIDON_ERROR("Unknown IP protocol: family = ", family);
		DEBUG_THROW(Exception, sslit("Unknown IP protocol"));
	}
}
bool SockAddr::is_private() const {
	const int family = get_family();
	switch(family){
	case AF_INET: {
		const ::sockaddr_in &sin = as_sin(m_data);
		DEBUG_THROW_UNLESS(m_size >= sizeof(sin), Exception, sslit("Invalid IPv4 SockAddr"));
		const unsigned char *const addr = reinterpret_cast<const unsigned char *>(&(sin.sin_addr));
		return is_ipv4_private(addr); }
	case AF_INET6: {
		const ::sockaddr_in6 &sin6 = as_sin6(m_data);
		DEBUG_THROW_UNLESS(m_size >= sizeof(sin6), Exception, sslit("Invalid IPv6 SockAddr"));
		const unsigned char *const addr = reinterpret_cast<const unsigned char *>(&(sin6.sin6_addr));
		if(are_zeroes(addr, 15)){
			if(addr[15] == 0){ // ::/128: 未指定的地址
				return true;
			} else if(addr[15] == 1){ // ::1/128: 回环地址
				return true;
			} else {
				return false;
			}
		} else if(are_zeroes(addr, 10) && (addr[10] == 0xFF) && (addr[11] == 0xFF)){ // IPv4 翻译地址
			return is_ipv4_private(addr + 12);
		} else if((addr[0] == 0x01) && are_zeroes(addr + 1, 7)){ // 100::/64 黑洞地址
			return true;
		} else if(addr[0] >= 0xFC){ // 私有地址、链路本地地址和广播地址
			return true;
		}
		return false; }
	default:
		LOG_POSEIDON_ERROR("Unknown IP protocol: family = ", family);
		DEBUG_THROW(Exception, sslit("Unknown IP protocol"));
	}
}
bool SockAddr::is_multicast() const {
	const int family = get_family();
	switch(family){
	case AF_INET: {
		const ::sockaddr_in &sin = as_sin(m_data);
		DEBUG_THROW_UNLESS(m_size >= sizeof(sin), Exception, sslit("Invalid IPv4 SockAddr"));
		const unsigned char *const addr = reinterpret_cast<const unsigned char *>(&(sin.sin_addr));
		return is_ipv4_multicast(addr); }
	case AF_INET6: {
		const ::sockaddr_in6 &sin6 = as_sin6(m_data);
		DEBUG_THROW_UNLESS(m_size >= sizeof(sin6), Exception, sslit("Invalid IPv6 SockAddr"));
		const unsigned char *const addr = reinterpret_cast<const unsigned char *>(&(sin6.sin6_addr));
		if((addr[0] == 0xFF) && (addr[1] == 0)){
			return true;
		} else if(are_zeroes(addr, 10) && (addr[10] == 0xFF) && (addr[11] == 0xFF)){ // IPv4 翻译地址
			return is_ipv4_multicast(addr + 12);
		}
		return false; }
	default:
		LOG_POSEIDON_ERROR("Unknown IP protocol: family = ", family);
		DEBUG_THROW(Exception, sslit("Unknown IP protocol"));
	}
}

}
