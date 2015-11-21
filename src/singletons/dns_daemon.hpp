// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2015, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_SINGLETONS_DNS_DAEMON_HPP_
#define POSEIDON_SINGLETONS_DNS_DAEMON_HPP_

#include <boost/shared_ptr.hpp>
#include <string>

namespace Poseidon {

class SockAddr;
class JobPromise;

struct DnsDaemon {
	static void start();
	static void stop();

	// 同步接口。
	static SockAddr sync_lookup(std::string host, unsigned port);

	// 异步接口。
	// 第一个参数是出参。
	static boost::shared_ptr<const JobPromise> async_lookup(boost::shared_ptr<SockAddr> sock_addr, std::string host, unsigned port);

private:
	DnsDaemon();
};

}

#endif
