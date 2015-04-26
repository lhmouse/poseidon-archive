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
	const void *getData() const {
		return m_data;
	}
	unsigned getSize() const {
		return m_size;
	}

	int getFamily() const;
};

extern IpPort getIpPortFromSockAddr(const SockAddr &sa);
extern SockAddr getSockAddrFromIpPort(const IpPort &addr);

}

#endif
