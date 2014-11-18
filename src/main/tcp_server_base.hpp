// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_TCP_SERVER_BASE_HPP_
#define POSEIDON_TCP_SERVER_BASE_HPP_

#include <boost/noncopyable.hpp>
#include <boost/shared_ptr.hpp>
#include "socket_server_base.hpp"

namespace Poseidon {

class TcpSessionBase;

// 抽象工厂模式
class TcpServerBase : public SocketServerBase {
private:
	class SslImplServer;
	class SslImplClient;

private:
	boost::scoped_ptr<SslImplServer> m_sslImplServer;

public:
	TcpServerBase(const IpPort &bindAddr, const char *cert, const char *privateKey);
	virtual ~TcpServerBase();

protected:
	// 工厂函数。返回空指针导致抛出一个异常。
	virtual boost::shared_ptr<TcpSessionBase> onClientConnect(ScopedFile client) const = 0;

public:
	bool poll() const;
};

}

#endif
