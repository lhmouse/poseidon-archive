// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2018, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_UDP_CLIENT_BASE_HPP_
#define POSEIDON_UDP_CLIENT_BASE_HPP_

#include "udp_session_base.hpp"

namespace Poseidon {

class UdpClientBase : public UdpSessionBase {
public:
	explicit UdpClientBase(const SockAddr &addr);
	~UdpClientBase();
};

}

#endif
