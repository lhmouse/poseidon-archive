// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_SINGLETONS_HTTP_SERVLET_MANAGER_HPP_
#define POSEIDON_SINGLETONS_HTTP_SERVLET_MANAGER_HPP_

#include <string>
#include <boost/shared_ptr.hpp>
#include <boost/weak_ptr.hpp>
#include <boost/function.hpp>
#include "../shared_ntmbs.hpp"
#include "../http/status.hpp"
#include "../http/request.hpp"
#include "../optional_map.hpp"
#include "../stream_buffer.hpp"

namespace Poseidon {

class HttpServlet;
class HttpSession;

typedef boost::function<
	void (boost::shared_ptr<HttpSession> hs, HttpRequest request)
	> HttpServletCallback;

struct HttpServletManager {
	static void start();
	static void stop();

	static std::size_t getMaxRequestLength();
	static unsigned long long getRequestTimeout();
	static unsigned long long getKeepAliveTimeout();

	// 返回的 shared_ptr 是该响应器的唯一持有者。
	static boost::shared_ptr<HttpServlet> registerServlet(
		std::size_t category, SharedNtmbs uri, HttpServletCallback callback);

	static boost::shared_ptr<const HttpServletCallback> getServlet(
		std::size_t category, const SharedNtmbs &uri);

private:
	HttpServletManager();
};

}

#endif
