#ifndef POSEIDON_SINGLETONS_EPOLL_DAEMON_HPP_
#define POSEIDON_SINGLETONS_EPOLL_DAEMON_HPP_

#include <boost/shared_ptr.hpp>

namespace Poseidon {

class TcpSessionBase;

struct EpollDaemon {
	static void start();
	static void stop();

	static void addSession(const boost::shared_ptr<TcpSessionBase> &session);
	static void touchSession(const boost::shared_ptr<TcpSessionBase> &session);

	// 返回的 shared_ptr 是该响应器的唯一持有者。
	static boost::shared_ptr<class PlayerServer> registerPlayerServer(
		const std::string &bindAddr, unsigned bindPort,
		const std::string &cert, const std::string &privateKey);
	static boost::shared_ptr<class HttpServer> registerHttpServer(
		const std::string &bindAddr, unsigned bindPort,
		const std::string &cert, const std::string &privateKey, const std::vector<std::string> &auth);

private:
	EpollDaemon();
};

}

#endif
