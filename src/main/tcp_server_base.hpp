// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_TCP_SERVER_BASE_HPP_
#define POSEIDON_TCP_SERVER_BASE_HPP_

#include <boost/shared_ptr.hpp>
#include "socket_server_base.hpp"

namespace Poseidon {

class ServerSslFactory;
class TcpSessionBase;

// 抽象工厂模式
class TcpServerBase : public SocketServerBase {
private:
	boost::scoped_ptr<ServerSslFactory> m_sslFactory;

public:
	TcpServerBase(const IpPort &bindAddr, const char *cert, const char *privateKey);
	virtual ~TcpServerBase();

protected:
	// 工厂函数。返回空指针导致抛出一个异常。
	virtual boost::shared_ptr<TcpSessionBase> onClientConnect(UniqueFile client) const = 0;

public:
	bool poll() const;
};

}

#endif
