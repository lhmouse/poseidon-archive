#ifndef POSEIDON_HTTP_WEBSOCKET_OPCODE_HPP_
#define POSEIDON_HTTP_WEBSOCKET_OPCODE_HPP_

namespace Poseidon {

enum WebSocketOpCode {
	WS_INVALID_OPCODE	= -1,
	WS_CONTINUATION		= 0x00,
	WS_DATA_TEXT		= 0x01,
	WS_DATA_BIN			= 0x02,
	WS_CLOSE			= 0x08,
	WS_PING				= 0x09,
	WS_PONG				= 0x0A,
};

enum {
	WS_FL_FIN			= 0x80,
	WS_FL_RSV1			= 0x40,
	WS_FL_RSV2			= 0x20,
	WS_FL_RSV3			= 0x10,
	WS_FL_CONTROL		= 0x08,
	WS_FL_OPCODE		= 0x0F,
};

}

#endif
