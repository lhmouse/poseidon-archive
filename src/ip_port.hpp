// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2017, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_IP_PORT_HPP_
#define POSEIDON_IP_PORT_HPP_

#include "cxx_ver.hpp"
#include <cstddef>

namespace Poseidon {

class SockAddr;

class IpPort {
private:
	char m_ip[128];
	boost::uint16_t m_port;

public:
	IpPort();
	IpPort(const char *ip_str, boost::uint16_t port_num);
	IpPort(const SockAddr &sock_addr);
	IpPort(const IpPort &ip_port) NOEXCEPT;
	IpPort &operator=(const IpPort &ip_port) NOEXCEPT;

public:
	const char *ip() const {
		return m_ip;
	}
	boost::uint16_t port() const {
		return m_port;
	}
};

extern const IpPort &unknown_ip_port() NOEXCEPT;
extern const IpPort &listening_ip_port() NOEXCEPT;

extern bool operator<(const IpPort &lhs, const IpPort &rhs) NOEXCEPT;

extern std::ostream &operator<<(std::ostream &os, const IpPort &rhs);

}

#endif
