// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2015, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_SOCK_ADDR_HPP_
#define POSEIDON_SOCK_ADDR_HPP_

#include <cstddef>

namespace Poseidon {

class IpPort;

class SockAddr {
private:
	char m_data[128];
	unsigned m_size;

public:
	SockAddr();
	SockAddr(const void *data, unsigned size);

public:
	const void *get_data() const {
		return m_data;
	}
	unsigned get_size() const {
		return m_size;
	}

	int get_family() const;
	// 如果是 IPv4 地址返回 false，如果是 IPv6 地址返回 true，否则抛出一个异常。
	bool is_ipv6() const;

	bool is_private() const;
};

extern IpPort get_ip_port_from_sock_addr(const SockAddr &sa);
extern SockAddr get_sock_addr_from_ip_port(const IpPort &addr);

}

#endif
