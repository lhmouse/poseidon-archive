// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2018, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_SYSTEM_HTTP_SERVER_HPP_
#define POSEIDON_SYSTEM_HTTP_SERVER_HPP_

#include <boost/shared_ptr.hpp>
#include <boost/function.hpp>
#include <boost/container/vector.hpp>

namespace Poseidon {

class System_http_servlet_base;

class System_http_server {
private:
	System_http_server();

public:
	static void start();
	static void stop();

	static boost::shared_ptr<const System_http_servlet_base> get_servlet(const char *uri);
	static void get_all_servlets(boost::container::vector<boost::shared_ptr<const System_http_servlet_base> > &ret);

	// 返回的 shared_ptr 是该处理程序的唯一持有者。
	static boost::shared_ptr<const System_http_servlet_base> register_servlet(boost::shared_ptr<System_http_servlet_base> servlet);
};

}

#endif
