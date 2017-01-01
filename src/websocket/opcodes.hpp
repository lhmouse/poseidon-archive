// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2017, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_WEBSOCKET_OPCODES_HPP_
#define POSEIDON_WEBSOCKET_OPCODES_HPP_

namespace Poseidon {

namespace WebSocket {
	typedef int OpCode;

	namespace OpCodes {
		enum {
			OP_INVALID          = -1,
			OP_CONTINUATION     = 0x00,
			OP_DATA_TEXT        = 0x01,
			OP_DATA_BINARY      = 0x02,
			OP_CLOSE            = 0x08,
			OP_PING             = 0x09,
			OP_PONG             = 0x0A,
		};

		enum {
			OP_FL_FIN           = 0x80,
			OP_FL_RSV1          = 0x40,
			OP_FL_RSV2          = 0x20,
			OP_FL_RSV3          = 0x10,
			OP_FL_CONTROL       = 0x08,
			OP_FL_OPCODE        = 0x0F,
		};
	}

	using namespace OpCodes;
}

}

#endif
