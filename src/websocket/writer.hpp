// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2015, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_WEBSOCKET_WRITER_HPP_
#define POSEIDON_WEBSOCKET_WRITER_HPP_

#include <string>
#include <boost/cstdint.hpp>
#include "status_codes.hpp"
#include "../stream_buffer.hpp"

namespace Poseidon {

namespace WebSocket {
	class Writer {
	public:
		Writer();
		virtual ~Writer();

	protected:
		virtual long onEncodedDataAvail(StreamBuffer encoded) = 0;

	public:
		long putMessage(int opcode, bool masked, StreamBuffer payload);
		long putCloseMessage(StatusCode statusCode, StreamBuffer additional);
	};
}

}

#endif
