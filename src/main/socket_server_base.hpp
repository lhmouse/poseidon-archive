#ifndef POSEIDON_SOCKET_SERVER_BASE_HPP_
#define POSEIDON_SOCKET_SERVER_BASE_HPP_

#include <string>
#include <boost/noncopyable.hpp>
#include <boost/enable_shared_from_this.hpp>
#include "raii.hpp"

namespace Poseidon {

class SocketServerBase : boost::noncopyable
	, public boost::enable_shared_from_this<SocketServerBase>
{
private:
	static bool tryAccept(boost::shared_ptr<const SocketServerBase> server);

private:
	const std::string m_bindAddr;
	volatile bool m_running;
	ScopedFile m_listen;

public:
	explicit SocketServerBase(const std::string &bindAddr);
	virtual ~SocketServerBase();

public:
	void start();
	void stop();

	// 如果该成员函数返回空指针，连接会被立即挂断。
	virtual boost::shared_ptr<class TcpPeer> onClientConnected(ScopedFile &client) const = 0;
};

}

#endif
