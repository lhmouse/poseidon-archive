// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2015, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_IP_PORT_HPP_
#define POSEIDON_IP_PORT_HPP_

#include "cxx_ver.hpp"
#include <iosfwd>
#include <string>
#include <utility>
#include "shared_nts.hpp"

namespace Poseidon {

struct IpPort {
	SharedNts ip;
	unsigned port;

	IpPort()
		: ip(), port()
	{
	}
	IpPort(SharedNts ip_, unsigned port_)
		: ip(STD_MOVE(ip_)), port(port_)
	{
	}

	void swap(IpPort &rhs) NOEXCEPT {
		using std::swap;
		swap(ip, rhs.ip);
		swap(port, rhs.port);
	}
};

inline void swap(IpPort &lhs, IpPort &rhs) NOEXCEPT {
	lhs.swap(rhs);
}

extern std::ostream &operator<<(std::ostream &os, const IpPort &rhs);

extern IpPort get_remote_ip_port_from_fd(int fd);
extern IpPort get_local_ip_port_from_fd(int fd);

}

#endif
