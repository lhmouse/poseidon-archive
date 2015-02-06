// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2015, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_HTTP_WEBSOCKET_STATUS_HPP_
#define POSEIDON_HTTP_WEBSOCKET_STATUS_HPP_

namespace Poseidon {

namespace WebSocketStatusCodes {
	typedef unsigned WebSocketStatus;

	enum {
		WS_NORMAL_CLOSURE		= 1000,
		WS_GOING_AWAY			= 1001,
		WS_PROTOCOL_ERROR		= 1002,
		WS_INACCEPTABLE			= 1003,
		WS_RESERVED_UNKNOWN		= 1004,
		WS_RESERVED_NO_STATUS	= 1005,
		WS_RESERVED_ABNORMAL	= 1006,
		WS_INCONSISTENT			= 1007,
		WS_ACCESS_DENIED		= 1008,
		WS_MESSAGE_TOO_LARGE	= 1009,
		WS_EXTENSION_NOT_AVAIL	= 1010,
		WS_INTERNAL_ERROR		= 1011,
		WS_RESERVED_TLS			= 1015,
	};
}

using namespace WebSocketStatusCodes;

}

#endif
