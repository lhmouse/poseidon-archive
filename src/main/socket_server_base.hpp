#ifndef POSEIDON_SOCKET_SERVER_BASE_HPP_
#define POSEIDON_SOCKET_SERVER_BASE_HPP_

#include <string>
#include <boost/noncopyable.hpp>
#include <boost/enable_shared_from_this.hpp>
#include "raii.hpp"

namespace Poseidon {

class TcpPeer;

// 抽象工厂模式
class SocketServerBase : boost::noncopyable
	, public boost::enable_shared_from_this<SocketServerBase>
{
private:
	static bool tryAccept(boost::shared_ptr<const SocketServerBase> server);

private:
	std::string m_bindAddr;
	volatile bool m_running;
	ScopedFile m_listen;

public:
	SocketServerBase(const std::string &bindAddr, unsigned bindPort);
	virtual ~SocketServerBase();

public:
	void start();
	void stop();

	// 工厂函数。
	// 如果该成员函数返回空指针，连接会被立即挂断。
	virtual boost::shared_ptr<TcpPeer> onClientConnect(ScopedFile &client) const = 0;
};

}

#endif
