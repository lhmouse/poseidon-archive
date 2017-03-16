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
	std::memset(m_data, 0, sizeof(sa_family_t));
	m_size = 0;
}
SockAddr::SockAddr(const void *data, std::size_t size){
	if(size > sizeof(m_data)){
		LOG_POSEIDON_ERROR("SockAddr size too large: size = ", size);
		DEBUG_THROW(Exception, sslit("SockAddr size too large"));
	}
	std::memcpy(m_data, data, size);
	m_size = size;
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
	const AUTO(ip, m_data + sizeof(::sa_family_t));
	if(family == AF_INET){
		return is_ipv4_private(ip);
	} else if(family == AF_INET6){
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

SockAddr get_sock_addr_from_ip_port(const IpPort &addr){
	::sockaddr_in sin;
	::sockaddr_in6 sin6;
	if(::inet_pton(AF_INET, addr.get_ip(), &(sin.sin_addr)) == 1){
		sin.sin_family = AF_INET;
		store_be(sin.sin_port, addr.get_port());
		return SockAddr(&sin, sizeof(sin));
	} else if(::inet_pton(AF_INET6, addr.get_ip(), &(sin6.sin6_addr)) == 1){
		sin6.sin6_family = AF_INET6;
		store_be(sin6.sin6_port, addr.get_port());
		return SockAddr(&sin6, sizeof(sin6));
	} else {
		LOG_POSEIDON_ERROR("Unknown address format: ip = ", addr.get_ip());
		DEBUG_THROW(Exception, sslit("Unknown address format"));
	}
}

}
