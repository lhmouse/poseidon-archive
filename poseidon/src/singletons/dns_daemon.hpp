// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2018, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_SINGLETONS_DNS_DAEMON_HPP_
#define POSEIDON_SINGLETONS_DNS_DAEMON_HPP_

#include "../sock_addr.hpp"
#include "../promise.hpp"
#include <boost/shared_ptr.hpp>
#include <boost/cstdint.hpp>
#include <string>

namespace Poseidon {

extern template class Promise_container<Sock_addr>;

class Dns_daemon {
private:
	Dns_daemon();

public:
	static void start();
	static void stop();

	// 同步接口。
	static Sock_addr look_up(const std::string &host, std::uint16_t port, bool prefer_ipv4 = true);

	// 异步接口。
	static boost::shared_ptr<const Promise_container<Sock_addr> > enqueue_for_looking_up(std::string host, std::uint16_t port, bool prefer_ipv4 = true);
};

}

#endif
