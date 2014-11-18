// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_SINGLETONS_EPOLL_DAEMON_HPP_
#define POSEIDON_SINGLETONS_EPOLL_DAEMON_HPP_

#include <boost/shared_ptr.hpp>
#include "../shared_ntmbs.hpp"
#include "../ip_port.hpp"

namespace Poseidon {

struct EpollSnapshotItem {
	IpPort remote;
	IpPort local;
	unsigned long long usOnline;
};

class TcpSessionBase;

struct EpollDaemon {
	static void start();
	static void stop();

	static std::vector<EpollSnapshotItem> snapshot();

	static void addSession(const boost::shared_ptr<TcpSessionBase> &session);
	static void touchSession(const boost::shared_ptr<TcpSessionBase> &session);

	// 返回的 shared_ptr 是该响应器的唯一持有者。
	static boost::shared_ptr<class PlayerServer> registerPlayerServer(std::size_t category,
		const IpPort &bindAddr, const char *cert, const char *privateKey);
	static boost::shared_ptr<class HttpServer> registerHttpServer(std::size_t category,
		const IpPort &bindAddr, const char *cert, const char *privateKey,
		const std::vector<std::string> &auth);

private:
	EpollDaemon();
};

}

#endif
