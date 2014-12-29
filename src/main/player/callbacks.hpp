// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_PLAYER_CALLBACKS_HPP_
#define POSEIDON_PLAYER_CALLBACKS_HPP_

#include <boost/function.hpp>
#include <boost/shared_ptr.hpp>
#include "../stream_buffer.hpp"

namespace Poseidon {

class PlayerSession;

typedef boost::function<
	void (boost::shared_ptr<PlayerSession> ps, StreamBuffer incoming)
	> PlayerServletCallback;

}

#endif
