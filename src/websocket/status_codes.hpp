// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2017, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_WEBSOCKET_STATUS_CODES_HPP_
#define POSEIDON_WEBSOCKET_STATUS_CODES_HPP_

namespace Poseidon {

namespace WebSocket {
	typedef unsigned StatusCode;

	namespace StatusCodes {
		enum {
			ST_NORMAL_CLOSURE       = 1000,
			ST_GOING_AWAY           = 1001,
			ST_PROTOCOL_ERROR       = 1002,
			ST_INACCEPTABLE         = 1003,
			ST_RESERVED_UNKNOWN     = 1004,
			ST_RESERVED_NO_STATUS   = 1005,
			ST_RESERVED_ABNORMAL    = 1006,
			ST_INCONSISTENT         = 1007,
			ST_ACCESS_DENIED        = 1008,
			ST_MESSAGE_TOO_LARGE    = 1009,
			ST_EXTENSION_NOT_AVAIL  = 1010,
			ST_INTERNAL_ERROR       = 1011,
			ST_RESERVED_TLS         = 1015,
		};
	}

	using namespace StatusCodes;
}

}

#endif
