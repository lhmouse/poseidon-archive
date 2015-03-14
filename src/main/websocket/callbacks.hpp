// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2015, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_WEBSOCKET_CALLBACKS_HPP_
#define POSEIDON_WEBSOCKET_CALLBACKS_HPP_

#include <boost/function.hpp>
#include <boost/shared_ptr.hpp>
#include "opcodes.hpp"
#include "../stream_buffer.hpp"

namespace Poseidon {

namespace WebSocket {
	class Session;

	typedef boost::function<
		void (const boost::shared_ptr<Session> &session, OpCode opcode, StreamBuffer incoming)
		> ServletCallback;
}

}

#endif
