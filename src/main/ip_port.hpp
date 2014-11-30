// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_IP_PORT_HPP_
#define POSEIDON_IP_PORT_HPP_

#include "cxx_ver.hpp"
#include <iosfwd>
#include "shared_ntmbs.hpp"

namespace Poseidon {

struct IpPort {
	SharedNtmbs ip;
	unsigned port;

	IpPort()
		: ip(), port()
	{
	}
	IpPort(SharedNtmbs ip_, unsigned port_, bool owning = false)
		: ip(STD_MOVE(ip_), owning), port(port_)
	{
	}
	IpPort(const IpPort &rhs, bool owning)
		: ip(rhs.ip, owning), port(rhs.port)
	{
	}
	IpPort(Move<IpPort> rhs, bool owning){
		rhs.swap(*this);
		if(owning){
			ip.forkOwning();
		}
	}

	void swap(IpPort &rhs) NOEXCEPT {
		ip.swap(rhs.ip);
		std::swap(port, rhs.port);
	}
};

inline void swap(IpPort &lhs, IpPort &rhs) NOEXCEPT {
	lhs.swap(rhs);
}

extern std::ostream &operator<<(std::ostream &os, const IpPort &rhs);
extern std::wostream &operator<<(std::wostream &os, const IpPort &rhs);

extern IpPort getRemoteIpPortFromFd(int fd);
extern IpPort getLocalIpPortFromFd(int fd);

}

#endif
