#ifndef POSEIDON_HTTP_SERVLET_MANAGER_HPP_
#define POSEIDON_HTTP_SERVLET_MANAGER_HPP_

#include <string>
#include <boost/shared_ptr.hpp>
#include <boost/weak_ptr.hpp>
#include <boost/function.hpp>
#include "http_status.hpp"
#include "http_verb.hpp"
#include "optional_map.hpp"

namespace Poseidon {

struct HttpServlet;

struct HttpServletManager {
	typedef boost::function<HttpStatus (
		OptionalMap &headers, std::string &contents,
		HttpVerb verb, const OptionalMap &getParams, const OptionalMap &postParams
	)> HttpServletCallback;

	// 返回的 shared_ptr 是该响应器的唯一持有者。
	static boost::shared_ptr<HttpServlet> registerServlet(const std::string &uri,
		boost::weak_ptr<void> dependency, HttpServletCallback callback);

	static boost::shared_ptr<const HttpServletCallback> getServlet(const std::string &uri);

private:
	HttpServletManager();
};

}

#endif
