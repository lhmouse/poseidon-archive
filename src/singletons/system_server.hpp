// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2017, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_SYSTEM_SERVER_HPP_
#define POSEIDON_SYSTEM_SERVER_HPP_

#include <boost/shared_ptr.hpp>
#include <boost/function.hpp>
#include <vector>

namespace Poseidon {

class SystemServletBase;

class SystemServer {
private:
	SystemServer();

public:
	static void start();
	static void stop();

	static boost::shared_ptr<const SystemServletBase> get_servlet(const char *uri);
	static void get_all_servlets(std::vector<boost::shared_ptr<const SystemServletBase> > &ret);

	// 返回的 shared_ptr 是该处理程序的唯一持有者。
	static boost::shared_ptr<const SystemServletBase> register_servlet(boost::shared_ptr<SystemServletBase> servlet);
};

}

#endif
