// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2017, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_IP_PORT_HPP_
#define POSEIDON_IP_PORT_HPP_

#include <cstddef>

namespace Poseidon {

class SockAddr;

class IpPort {
private:
	char m_ip[128];
	unsigned m_port;

public:
	IpPort();
	IpPort(const char *ip, unsigned port);

public:
	const char *get_ip() const {
		return m_ip;
	}
	unsigned get_port() const {
		return m_port;
	}
};

extern const IpPort &unknown_ip_port() NOEXCEPT;

extern std::ostream &operator<<(std::ostream &os, const IpPort &rhs);

extern IpPort get_ip_port_from_sock_addr(const SockAddr &sa);

}

#endif
