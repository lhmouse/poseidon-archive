// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2017, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_TCP_SERVER_BASE_HPP_
#define POSEIDON_TCP_SERVER_BASE_HPP_

#include <boost/shared_ptr.hpp>
#include <boost/scoped_ptr.hpp>
#include "socket_base.hpp"
#include "sock_addr.hpp"
#include "ip_port.hpp"

namespace Poseidon {

class ServerSslFactory;
class TcpSessionBase;

// 抽象工厂模式
class TcpServerBase : public SocketBase {
private:
	boost::scoped_ptr<ServerSslFactory> m_ssl_factory;

public:
	explicit TcpServerBase(const SockAddr &addr, const char *certificate = "", const char *private_key = "");
	~TcpServerBase();

protected:
	// 工厂函数。返回空指针导致抛出一个异常。
	virtual boost::shared_ptr<TcpSessionBase> on_client_connect(Move<UniqueFile> client) = 0;

public:
	int poll_read_and_process(unsigned char *hint_buffer, std::size_t hint_capacity, bool readable) OVERRIDE;
};

}

#endif
