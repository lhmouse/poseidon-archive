#ifndef POSEIDON_TCP_SERVER_HPP_
#define POSEIDON_TCP_SERVER_HPP_

#include <string>
#include <boost/noncopyable.hpp>
#include <boost/scoped_ptr.hpp>
#include <boost/shared_ptr.hpp>
#include "raii.hpp"
#include "shared_ntmbs.hpp"

namespace Poseidon {

class TcpSessionBase;

// 抽象工厂模式
class TcpServerBase : boost::noncopyable {
private:
	class SslImplServer;
	class SslImplClient;

private:
	const std::string m_bindAddr;
	const unsigned m_bindPort;

	ScopedFile m_listen;
	boost::scoped_ptr<SslImplServer> m_sslImplServer;

public:
	TcpServerBase(std::string bindAddr, unsigned bindPort,
		const SharedNtmbs &cert, const SharedNtmbs &privateKey);
	virtual ~TcpServerBase();

protected:
	// 工厂函数。返回空指针导致抛出一个异常。
	virtual boost::shared_ptr<TcpSessionBase> onClientConnect(ScopedFile client) const = 0;

public:
	const std::string &getLocalIp() const {
		return m_bindAddr;
	}
	unsigned getLocalPort() const {
		return m_bindPort;
	}

	boost::shared_ptr<TcpSessionBase> tryAccept() const;
};

}

#endif
