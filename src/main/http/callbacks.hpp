// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2015, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_HTTP_CALLBACKS_HPP_
#define POSEIDON_HTTP_CALLBACKS_HPP_

#include <boost/function.hpp>
#include <boost/shared_ptr.hpp>

namespace Poseidon {

namespace Http {
	class Session;
	class Request;

	typedef boost::function<
		void (const boost::shared_ptr<Session> &session, const Request &request)
		> ServletCallback;
}

}

#endif
