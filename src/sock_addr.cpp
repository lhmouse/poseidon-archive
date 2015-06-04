// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2015, LH_Mouse. All wrongs reserved.

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

int SockAddr::getFamily() const {
	const AUTO(p, reinterpret_cast<const ::sockaddr *>(m_data));
	if(m_size < sizeof(p->sa_family)){
		LOG_POSEIDON_ERROR("Invalid SockAddr: size = ", m_size);
		DEBUG_THROW(Exception, sslit("Invalid SockAddr"));
	}
	return p->sa_family;
}
bool SockAddr::isIpv6() const {
	const int family = getFamily();
	if(family == AF_INET){
		return false;
	} else if(family == AF_INET6){
		return true;
	}

	LOG_POSEIDON_WARNING("Unknown IP protocol ", family);
	DEBUG_THROW(Exception, sslit("Unknown IP protocol"));
}

bool SockAddr::isPrivate() const {
	const int family = getFamily();
	if(family == AF_INET){
		const AUTO_REF(ip, reinterpret_cast<const unsigned char (&)[4]>(
			static_cast<const ::sockaddr_in *>(getData())->sin_addr));
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
		}
		return false;
	} else if(family == AF_INET6){
		static const unsigned char ZEROES[16] = { };

		const AUTO_REF(ip, reinterpret_cast<const unsigned char (&)[16]>(
			static_cast<const ::sockaddr_in6 *>(getData())->sin6_addr));
		if(std::memcmp(ip, ZEROES, 15) == 0){
			if(ip[15] == 0){ // ::/128: 未指定的地址
				return true;
			}
			if(ip[15] == 1){ // ::1/128: 回环地址
				return true;
			}
		} else if((std::memcmp(ip, ZEROES, 10) == 0) && (ip[10] == 0xFF) && (ip[11] == 0xFF)){ // IPv4 翻译地址
			if(ip[12] == 0){ // 0.0.0.0/8: 当前网络地址
				return true;
			} else if(ip[12] == 10){ // 10.0.0.0/8: A 类私有地址
				return true;
			} else if(ip[12] == 127){ // 127.0.0.0/8: 回环地址
				return true;
			} else if((ip[12] == 172) && ((ip[13] & 0xF0) == 16)){ // 172.16.0.0/12: B 类私有地址
				return true;
			} else if((ip[12] == 169) && (ip[13] == 254)){ // 169.254.0.0/16: 链路本地地址
				return true;
			} else if((ip[12] == 192) && (ip[13] == 168)){ // 192.168.0.0/16: C 类私有地址
				return true;
			} else if(ip[12] >= 224){ // D 类、E 类地址和广播地址
				return true;
			}
		} else if((ip[0] == 0x01) && (std::memcmp(ip + 1, ZEROES, 7) == 0)){ // 100::/64 黑洞地址
			return true;
		} else if(ip[0] >= 0xFC){ // 私有地址、链路本地地址和广播地址
			return true;
		}
		return false;
	}

	LOG_POSEIDON_WARNING("Unknown IP protocol ", family);
	DEBUG_THROW(Exception, sslit("Unknown IP protocol"));
}

IpPort getIpPortFromSockAddr(const SockAddr &sa){
	const int family = sa.getFamily();
	if(family == AF_INET){
		if(sa.getSize() < sizeof(::sockaddr_in)){
			LOG_POSEIDON_WARNING("Invalid IPv4 SockAddr: size = ", sa.getSize());
			DEBUG_THROW(Exception, sslit("Invalid IPv4 SockAddr"));
		}
		char ip[INET_ADDRSTRLEN];
		const char *const str = ::inet_ntop(AF_INET,
			&static_cast<const ::sockaddr_in *>(sa.getData())->sin_addr, ip, sizeof(ip));
		if(!str){
			DEBUG_THROW(SystemException);
		}
		return IpPort(SharedNts(str),
			loadBe(static_cast<const ::sockaddr_in *>(sa.getData())->sin_port));
	} else if(family == AF_INET6){
		if(sa.getSize() < sizeof(::sockaddr_in6)){
			LOG_POSEIDON_WARNING("Invalid IPv6 SockAddr: size = ", sa.getSize());
			DEBUG_THROW(Exception, sslit("Invalid IPv6 SockAddr"));
		}
		char ip[INET6_ADDRSTRLEN];
		const char *const str = ::inet_ntop(AF_INET6,
			&static_cast<const ::sockaddr_in6 *>(sa.getData())->sin6_addr, ip, sizeof(ip));
		if(!str){
			DEBUG_THROW(SystemException);
		}
		return IpPort(SharedNts(str),
			loadBe(static_cast<const ::sockaddr_in6 *>(sa.getData())->sin6_port));
	}

	LOG_POSEIDON_WARNING("Unknown IP protocol ", family);
	DEBUG_THROW(Exception, sslit("Unknown IP protocol"));
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
	DEBUG_THROW(Exception, sslit("Unknown address format"));
}

}
