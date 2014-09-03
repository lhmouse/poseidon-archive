#ifndef POSEIDON_HTTP_SERVLET_MANAGER_HPP_
#define POSEIDON_HTTP_SERVLET_MANAGER_HPP_

#include <string>
#include <boost/shared_ptr.hpp>
#include <boost/weak_ptr.hpp>
#include <boost/function.hpp>
#include "../http_status.hpp"
#include "../http_verb.hpp"
#include "../optional_map.hpp"

namespace Poseidon {

class HttpServlet;

typedef boost::function<
	HttpStatus (OptionalMap &headers, std::string &contents,
		HttpVerb verb, const OptionalMap &getParams, const OptionalMap &postParams)
	> HttpServletCallback;

struct HttpServletManager {
	// 返回的 shared_ptr 是该响应器的唯一持有者。
	// callback 禁止 move，否则可能出现主模块中引用子模块内存的情况。
	static boost::shared_ptr<const HttpServlet> registerServlet(const std::string &uri,
		boost::weak_ptr<void> dependency, const HttpServletCallback &callback);

	static boost::shared_ptr<const HttpServletCallback> getServlet(const std::string &uri);

private:
	HttpServletManager();
};

}

#endif
