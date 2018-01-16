// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2018, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_UDP_SERVER_BASE_HPP_
#define POSEIDON_UDP_SERVER_BASE_HPP_

#include "udp_session_base.hpp"

namespace Poseidon {

class UdpServerBase : public UdpSessionBase {
public:
	explicit UdpServerBase(const SockAddr &addr);
	~UdpServerBase();
};

}

#endif
