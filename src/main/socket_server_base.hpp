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

private:
	bool tryAccept() const;

protected:
	// 工厂函数。
	// 如果该成员函数返回空指针，连接会被立即挂断。
	virtual boost::shared_ptr<TcpPeer> onClientConnect(ScopedFile &client) const = 0;

public:
	// 交由 epoll 调度。这里使用自己的 shared_ptr 附带转移了所有权。
	// 该对象将在 epoll 守护线程停止时被销毁。
	void handOver() const;
};

}

#endif
