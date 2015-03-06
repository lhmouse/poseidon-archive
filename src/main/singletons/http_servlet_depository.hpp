// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2015, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_SINGLETONS_HTTP_SERVLET_DEPOSITORY_HPP_
#define POSEIDON_SINGLETONS_HTTP_SERVLET_DEPOSITORY_HPP_

#include <string>
#include <boost/shared_ptr.hpp>
#include <boost/weak_ptr.hpp>
#include <boost/function.hpp>
#include "../shared_nts.hpp"
#include "../http/callbacks.hpp"
#include "../optional_map.hpp"
#include "../stream_buffer.hpp"

namespace Poseidon {

class HttpServlet;

struct HttpServletDepository {
	static void start();
	static void stop();

	static std::size_t getMaxRequestLength();
	static unsigned long long getKeepAliveTimeout();

	// 返回的 shared_ptr 是该响应器的唯一持有者。
	static boost::shared_ptr<HttpServlet> create(
		std::size_t category, SharedNts uri, HttpServletCallback callback);

	static boost::shared_ptr<const HttpServletCallback> get(
		std::size_t category, const char *uri);

private:
	HttpServletDepository();
};

}

#endif
