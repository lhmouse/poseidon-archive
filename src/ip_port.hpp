// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2018, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_IP_PORT_HPP_
#define POSEIDON_IP_PORT_HPP_

#include "cxx_ver.hpp"
#include <cstddef>

namespace Poseidon {

class Sock_addr;

class Ip_port {
private:
	char m_ip[128];
	boost::uint16_t m_port;

public:
	Ip_port();
	Ip_port(const char *ip_str, boost::uint16_t port_num);
	Ip_port(const Sock_addr &sock_addr);
	Ip_port(const Ip_port &ip_port) NOEXCEPT;
	Ip_port &operator=(const Ip_port &ip_port) NOEXCEPT;

public:
	const char *ip() const {
		return m_ip;
	}
	boost::uint16_t port() const {
		return m_port;
	}
};

extern const Ip_port &unknown_ip_port() NOEXCEPT;
extern const Ip_port &listening_ip_port() NOEXCEPT;

extern bool operator<(const Ip_port &lhs, const Ip_port &rhs) NOEXCEPT;

extern std::ostream &operator<<(std::ostream &os, const Ip_port &rhs);

}

#endif
