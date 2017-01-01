// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2017, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_SINGLETONS_EPOLL_DAEMON_HPP_
#define POSEIDON_SINGLETONS_EPOLL_DAEMON_HPP_

#include <boost/shared_ptr.hpp>
#include <boost/cstdint.hpp>
#include "../ip_port.hpp"

namespace Poseidon {

class TcpSessionBase;
class SocketServerBase;

class EpollDaemon {
private:
	EpollDaemon();

public:
	struct SnapshotElement {
		IpPort remote;
		IpPort local;
		boost::uint64_t ms_online;
	};

	static void start();
	static void stop();

	static boost::uint64_t get_tcp_request_timeout();

	static std::vector<SnapshotElement> snapshot();

	static void add_session(const boost::shared_ptr<TcpSessionBase> &session);
	static void register_server(boost::weak_ptr<const SocketServerBase> server);
};

}

#endif
