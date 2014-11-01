#ifndef POSEIDON_TCP_SERVER_HPP_
#define POSEIDON_TCP_SERVER_HPP_

#include <string>
#include <boost/noncopyable.hpp>
#include <boost/scoped_ptr.hpp>
#include <boost/shared_ptr.hpp>
#include "raii.hpp"
#include "shared_ntmbs.hpp"
#include "ip_port.hpp"

namespace Poseidon {

class TcpSessionBase;

// 抽象工厂模式
class TcpServerBase : boost::noncopyable {
private:
	class SslImplServer;
	class SslImplClient;

private:
	const IpPort m_localInfo;
	const ScopedFile m_listen;

	boost::scoped_ptr<SslImplServer> m_sslImplServer;

public:
	TcpServerBase(const IpPort &bindAddr, const char *cert, const char *privateKey);
	virtual ~TcpServerBase();

protected:
	// 工厂函数。返回空指针导致抛出一个异常。
	virtual boost::shared_ptr<TcpSessionBase> onClientConnect(ScopedFile client) const = 0;

public:
	const IpPort &getLocalInfo() const {
		return m_localInfo;
	}

	boost::shared_ptr<TcpSessionBase> tryAccept() const;
};

}

#endif
