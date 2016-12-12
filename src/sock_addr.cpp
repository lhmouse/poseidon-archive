// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2016, LH_Mouse. All wrongs reserved.

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
}

SockAddr::SockAddr(){
	m_size = 0;
}
SockAddr::SockAddr(const void *data, unsigned size){
	if(size > sizeof(m_data)){
		LOG_POSEIDON_ERROR("SockAddr size too large: ", size);
		DEBUG_THROW(Exception, sslit("SockAddr size too large"));
	}
	m_size = size;
	std::memcpy(m_data, data, size);
}

int SockAddr::get_family() const {
	const AUTO(p, reinterpret_cast<const ::sockaddr *>(get_data()));
	if(m_size < sizeof(p->sa_family)){
		LOG_POSEIDON_ERROR("Invalid SockAddr: size = ", m_size);
		DEBUG_THROW(Exception, sslit("Invalid SockAddr"));
	}
	return p->sa_family;
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
	const AUTO(ip, reinterpret_cast<const unsigned char *>(&static_cast<const ::sockaddr_in *>(get_data())->sin_addr));
	if(family == AF_INET){
		return is_ipv4_private(ip);
	} else if(family == AF_INET6){
		static CONSTEXPR const unsigned char s_zeroes[16] = { };
		if(std::memcmp(ip, s_zeroes, 15) == 0){
			if(ip[15] == 0){ // ::/128: 未指定的地址
				return true;
			} else if(ip[15] == 1){ // ::1/128: 回环地址
				return true;
			} else {
				return false;
			}
		} else if((std::memcmp(ip, s_zeroes, 10) == 0) && (ip[10] == 0xFF) && (ip[11] == 0xFF)){ // IPv4 翻译地址
			return is_ipv4_private(ip + 12);
		} else if((ip[0] == 0x01) && (std::memcmp(ip + 1, s_zeroes, 7) == 0)){ // 100::/64 黑洞地址
			return true;
		} else if(ip[0] >= 0xFC){ // 私有地址、链路本地地址和广播地址
			return true;
		} else {
			return false;
		}
	} else {
		LOG_POSEIDON_WARNING("Unknown IP protocol ", family);
		DEBUG_THROW(Exception, sslit("Unknown IP protocol"));
	}
}

IpPort get_ip_port_from_sock_addr(const SockAddr &sa){
	const int family = sa.get_family();
	if(family == AF_INET){
		if(sa.get_size() < sizeof(::sockaddr_in)){
			LOG_POSEIDON_WARNING("Invalid IPv4 SockAddr: size = ", sa.get_size());
			DEBUG_THROW(Exception, sslit("Invalid IPv4 SockAddr"));
		}
		char ip[INET_ADDRSTRLEN];
		const char *const str = ::inet_ntop(AF_INET,
			&static_cast<const ::sockaddr_in *>(sa.get_data())->sin_addr, ip, sizeof(ip));
		if(!str){
			DEBUG_THROW(SystemException);
		}
		return IpPort(SharedNts(str),
			load_be(static_cast<const ::sockaddr_in *>(sa.get_data())->sin_port));
	} else if(family == AF_INET6){
		if(sa.get_size() < sizeof(::sockaddr_in6)){
			LOG_POSEIDON_WARNING("Invalid IPv6 SockAddr: size = ", sa.get_size());
			DEBUG_THROW(Exception, sslit("Invalid IPv6 SockAddr"));
		}
		char ip[INET6_ADDRSTRLEN];
		const char *const str = ::inet_ntop(AF_INET6,
			&static_cast<const ::sockaddr_in6 *>(sa.get_data())->sin6_addr, ip, sizeof(ip));
		if(!str){
			DEBUG_THROW(SystemException);
		}
		return IpPort(SharedNts(str),
			load_be(static_cast<const ::sockaddr_in6 *>(sa.get_data())->sin6_port));
	} else {
		LOG_POSEIDON_WARNING("Unknown IP protocol ", family);
		DEBUG_THROW(Exception, sslit("Unknown IP protocol"));
	}
}
SockAddr get_sock_addr_from_ip_port(const IpPort &addr){
	::sockaddr_in sin;
	::sockaddr_in6 sin6;
	if(::inet_pton(AF_INET, addr.ip.get(), &sin.sin_addr) == 1){
		sin.sin_family = AF_INET;
		store_be(sin.sin_port, addr.port);
		return SockAddr(&sin, sizeof(sin));
	} else if(::inet_pton(AF_INET6, addr.ip.get(), &sin6.sin6_addr) == 1){
		sin6.sin6_family = AF_INET6;
		store_be(sin6.sin6_port, addr.port);
		return SockAddr(&sin6, sizeof(sin6));
	} else {
		LOG_POSEIDON_ERROR("Unknown address format: ", addr.ip);
		DEBUG_THROW(Exception, sslit("Unknown address format"));
	}
}

}
