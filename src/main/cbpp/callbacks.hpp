// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2015, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_CBPP_CALLBACKS_HPP_
#define POSEIDON_CBPP_CALLBACKS_HPP_

#include <boost/function.hpp>
#include <boost/shared_ptr.hpp>
#include "../stream_buffer.hpp"

namespace Poseidon {

namespace Cbpp {
	class Session;

	typedef boost::function<
		void (boost::shared_ptr<Session> session, StreamBuffer incoming)
		> ServletCallback;
}

}

#endif
