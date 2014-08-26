#ifndef POSEIDON_EPOLL_DAEMON_HPP_
#define POSEIDON_EPOLL_DAEMON_HPP_

#include <boost/shared_ptr.hpp>
#include "../raii.hpp"

namespace Poseidon {

class TcpPeer;
class SocketServerBase;

struct EpollDaemon {
	static void start();
	static void stop();

	// 对于写事件采用延迟写入与写入合并策略。
	static void notifyWriteable(boost::shared_ptr<TcpPeer> peer);
	// 注册 TCP socket 服务器。这里收养了所有权。
	static void addSocketServer(boost::shared_ptr<SocketServerBase> server);

private:
	EpollDaemon();
};

}

#endif
