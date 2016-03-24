// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2016, LH_Mouse. All wrongs reserved.

#include "../precompiled.hpp"
#include "request_headers.hpp"
#include <string.h>

namespace Poseidon {

namespace Http {
	bool is_keep_alive_enabled(const RequestHeaders &request_headers) NOEXCEPT {
		const AUTO_REF(connection, request_headers.headers.get("Connection"));
		if(request_headers.version < 10001){
			return ::strcasecmp(connection.c_str(), "Keep-Alive") == 0;
		} else {
			return ::strcasecmp(connection.c_str(), "Close") != 0;
		}
	}
}

}
