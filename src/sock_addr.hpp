// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2017, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_SOCK_ADDR_HPP_
#define POSEIDON_SOCK_ADDR_HPP_

#include "cxx_ver.hpp"
#include <cstddef>

namespace Poseidon {

class IpPort;

class SockAddr {
private:
	unsigned char m_data[128];
	std::size_t m_size;

public:
	SockAddr();
	SockAddr(const void *addr_data, std::size_t addr_size);
	SockAddr(const IpPort &ip_port);
	SockAddr(const SockAddr &rhs) NOEXCEPT;
	SockAddr &operator=(const SockAddr &rhs) NOEXCEPT;

public:
	const void *data() const {
		return m_data;
	}
	std::size_t size() const {
		return m_size;
	}

	int get_family() const;
	// 如果是 IPv4 地址返回 false，如果是 IPv6 地址返回 true，否则抛出一个异常。
	bool is_ipv6() const;
	bool is_private() const;
};

}

#endif
