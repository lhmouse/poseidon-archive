// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2016, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_UDP_SERVER_BASE_HPP_
#define POSEIDON_UDP_SERVER_BASE_HPP_

#include <boost/shared_ptr.hpp>
#include "socket_server_base.hpp"

namespace Poseidon {

class UdpServerBase : public SocketServerBase {
public:
	explicit UdpServerBase(const IpPort &bind_addr);
	virtual ~UdpServerBase();

public:
	bool poll() const OVERRIDE;
};

}

#endif
