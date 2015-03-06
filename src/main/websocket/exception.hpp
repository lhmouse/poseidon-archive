// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2015, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_WEBSOCKET_EXCEPTION_HPP_
#define POSEIDON_WEBSOCKET_EXCEPTION_HPP_

#include "../protocol_exception.hpp"
#include "status_codes.hpp"

namespace Poseidon {

class WebSocketException : public ProtocolException {
public:
	WebSocketException(const char *file, std::size_t line, WebSocketStatus status, SharedNts message);
	~WebSocketException() NOEXCEPT;

public:
	WebSocketStatus status() const NOEXCEPT {
		return static_cast<WebSocketStatus>(code());
	}
};

}

#endif