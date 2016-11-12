// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2016, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_TCP_SERVER_BASE_HPP_
#define POSEIDON_TCP_SERVER_BASE_HPP_

#include <boost/shared_ptr.hpp>
#include "socket_server_base.hpp"
#include "sock_addr.hpp"

namespace Poseidon {

class ServerSslFactory;
class TcpSessionBase;

// 抽象工厂模式
class TcpServerBase : public SocketServerBase {
private:
	boost::scoped_ptr<ServerSslFactory> m_ssl_factory;

public:
	TcpServerBase(const SockAddr &addr, const char *cert, const char *private_key);
	TcpServerBase(const IpPort &addr, const char *cert, const char *private_key);
	~TcpServerBase();

private:
	void init_ssl_factory(const char *cert, const char *private_key);

protected:
	// 工厂函数。返回空指针导致抛出一个异常。
	virtual boost::shared_ptr<TcpSessionBase> on_client_connect(UniqueFile client) const = 0;

public:
	bool poll() const OVERRIDE;
};

}

#endif
