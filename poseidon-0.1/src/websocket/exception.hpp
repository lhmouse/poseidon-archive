// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2016, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_WEBSOCKET_EXCEPTION_HPP_
#define POSEIDON_WEBSOCKET_EXCEPTION_HPP_

#include "../protocol_exception.hpp"
#include "status_codes.hpp"

namespace Poseidon {

namespace WebSocket {
	class Exception : public ProtocolException {
	public:
		Exception(const char *file, std::size_t line, const char *func, StatusCode status_code, SharedNts message = SharedNts());
		~Exception() NOEXCEPT;

	public:
		StatusCode get_status_code() const NOEXCEPT {
			return static_cast<StatusCode>(get_code());
		}
	};
}

}

#endif