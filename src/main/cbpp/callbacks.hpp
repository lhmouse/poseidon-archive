// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2015, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_CBPP_CALLBACKS_HPP_
#define POSEIDON_CBPP_CALLBACKS_HPP_

#include <boost/function.hpp>
#include <boost/shared_ptr.hpp>
#include "../stream_buffer.hpp"

namespace Poseidon {

class CbppSession;

typedef boost::function<
	void (boost::shared_ptr<CbppSession> session, StreamBuffer incoming)
	> CbppServletCallback;

}

#endif
