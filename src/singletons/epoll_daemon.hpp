// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2017, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_SINGLETONS_EPOLL_DAEMON_HPP_
#define POSEIDON_SINGLETONS_EPOLL_DAEMON_HPP_

#include <boost/shared_ptr.hpp>
#include <boost/cstdint.hpp>
#include "../ip_port.hpp"

namespace Poseidon {

class SocketBase;

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

	static std::vector<SnapshotElement> snapshot();
	static void add_socket(const boost::shared_ptr<SocketBase> &socket);
	static bool mark_socket_writeable(int fd) NOEXCEPT;
};

}

#endif
