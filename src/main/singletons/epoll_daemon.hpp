#ifndef POSEIDON_EPOLL_DAEMON_HPP_
#define POSEIDON_EPOLL_DAEMON_HPP_

#include <boost/shared_ptr.hpp>
#include "../raii.hpp"

namespace Poseidon {

class TcpSessionBase;
class SocketServerBase;

struct EpollDaemon {
	static void start();
	static void stop();

	static void refreshSession(boost::shared_ptr<TcpSessionBase> session);
	// 注册 TCP socket 服务器。这里收养了所有权。
	static void addSocketServer(boost::shared_ptr<SocketServerBase> server);

private:
	EpollDaemon();
};

}

#endif
