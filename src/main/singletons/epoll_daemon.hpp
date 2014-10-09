#ifndef POSEIDON_SINGLETONS_EPOLL_DAEMON_HPP_
#define POSEIDON_SINGLETONS_EPOLL_DAEMON_HPP_

#include <boost/shared_ptr.hpp>

namespace Poseidon {

class TcpSessionBase;
class TcpServerBase;

struct EpollDaemon {
	static void start();
	static void stop();

	static void addSession(const boost::shared_ptr<TcpSessionBase> &session);
	static void refreshSession(const boost::shared_ptr<TcpSessionBase> &session);

	// 注册 TCP socket 服务器。这里拷贝了所有权。
	static void addTcpServer(boost::shared_ptr<const TcpServerBase> server);

private:
	EpollDaemon();
};

}

#endif
