#ifndef POSEIDON_SINGLETONS_HTTP_SERVLET_MANAGER_HPP_
#define POSEIDON_SINGLETONS_HTTP_SERVLET_MANAGER_HPP_

#include "../../cxx_ver.hpp"
#include <string>
#include <boost/shared_ptr.hpp>
#include <boost/weak_ptr.hpp>
#include "../http/status.hpp"
#include "../http/request.hpp"
#include "../optional_map.hpp"
#include "../stream_buffer.hpp"

#ifdef POSEIDON_CXX11
#   include <functional>
#else
#   include <tr1/functional>
#endif

namespace Poseidon {

class HttpServlet;
class HttpSession;

typedef TR1::function<
	void (boost::shared_ptr<HttpSession> hs, HttpRequest request)
	> HttpServletCallback;

struct HttpServletManager {
	static void start();
	static void stop();

	static std::size_t getMaxRequestLength();
	static unsigned long long getRequestTimeout();
	static unsigned long long getKeepAliveTimeout();

	// 返回的 shared_ptr 是该响应器的唯一持有者。
	// callback 禁止 move，否则可能出现主模块中引用子模块内存的情况。
	static boost::shared_ptr<HttpServlet> registerServlet(const std::string &uri,
		const boost::weak_ptr<const void> &dependency, const HttpServletCallback &callback);

	static boost::shared_ptr<const HttpServletCallback> getServlet(
		boost::shared_ptr<const void> &lockedDep, const std::string &uri);

private:
	HttpServletManager();
};

}

#endif
