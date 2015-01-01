// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2015, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_HTTP_CALLBACKS_HPP_
#define POSEIDON_HTTP_CALLBACKS_HPP_

#include <boost/function.hpp>
#include <boost/shared_ptr.hpp>
#include "request.hpp"

namespace Poseidon {

class HttpSession;

typedef boost::function<
	void (boost::shared_ptr<HttpSession> hs, HttpRequest request)
	> HttpServletCallback;

}

#endif
