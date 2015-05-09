// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2015, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_TCP_CLIENT_BASE_HPP_
#define POSEIDON_TCP_CLIENT_BASE_HPP_

#include "tcp_session_base.hpp"
#include <string>
#include <boost/shared_ptr.hpp>
#include "sock_addr.hpp"

namespace Poseidon {

class IpPort;

class TcpClientBase : private SockAddr, public TcpSessionBase {
protected:
	TcpClientBase(const SockAddr &addr, bool useSsl);
	TcpClientBase(const IpPort &addr, bool useSsl);
	~TcpClientBase();

private:
	void realConnect(bool useSsl);

protected:
	void onReadAvail(const void *data, std::size_t size) OVERRIDE = 0;

public:
	void goResident();
};

}

#endif
