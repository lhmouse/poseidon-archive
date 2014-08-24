#ifndef POSEIDON_SOCKET_SERVER_BASE_HPP_
#define POSEIDON_SOCKET_SERVER_BASE_HPP_

#include <string>
#include <boost/noncopyable.hpp>
#include "raii.hpp"
#include "virtual_shared_from_this.hpp"

namespace Poseidon {

class TcpPeer;

// 抽象工厂模式
class SocketServerBase : boost::noncopyable
	, public virtual VirtualSharedFromThis
{
private:
	std::string m_bindAddr;
	ScopedFile m_listen;

public:
	SocketServerBase(const std::string &bindAddr, unsigned bindPort);
	virtual ~SocketServerBase();

protected:
	// 工厂函数。
	// 如果该成员函数返回空指针，连接会被立即挂断。
	virtual boost::shared_ptr<TcpPeer> onClientConnect(ScopedFile &client) const = 0;

public:
	boost::shared_ptr<TcpPeer> tryAccept() const;
};

}

#endif
