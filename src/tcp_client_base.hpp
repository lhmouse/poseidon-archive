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

class TcpClientBase : protected SockAddr, public TcpSessionBase {
protected:
	TcpClientBase(const SockAddr &addr, bool use_ssl);
	TcpClientBase(const IpPort &addr, bool use_ssl);
	~TcpClientBase();

private:
	void real_connect(bool use_ssl);

protected:
	void on_read_avail(StreamBuffer data) OVERRIDE = 0;

public:
	void go_resident();
};

}

#endif
