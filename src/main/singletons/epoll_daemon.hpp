// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_SINGLETONS_EPOLL_DAEMON_HPP_
#define POSEIDON_SINGLETONS_EPOLL_DAEMON_HPP_

#include <boost/shared_ptr.hpp>
#include "../ip_port.hpp"

namespace Poseidon {

struct EpollSnapshotItem {
	IpPort remote;
	IpPort local;
	unsigned long long usOnline;
};

class TcpSessionBase;
class SocketServerBase;

struct EpollDaemon {
	static void start();
	static void stop();

	static std::vector<EpollSnapshotItem> snapshot();

	static void addSession(const boost::shared_ptr<TcpSessionBase> &session);
	static void touchSession(const boost::shared_ptr<TcpSessionBase> &session);

	static void registerServer(boost::shared_ptr<SocketServerBase> server);

private:
	EpollDaemon();
};

}

#endif
