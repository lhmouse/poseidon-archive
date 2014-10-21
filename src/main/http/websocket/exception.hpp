#ifndef POSEIDON_HTTP_WEBSOCKET_EXCEPTION_HPP_
#define POSEIDON_HTTP_WEBSOCKET_EXCEPTION_HPP_

#include "../../exception.hpp"
#include "status.hpp"

namespace Poseidon {

class WebSocketException : public ProtocolException {
public:
	WebSocketException(const char *file, std::size_t line, WebSocketStatus status,
		std::string reason = std::string())
		: ProtocolException(file, line, STD_MOVE(reason), static_cast<unsigned>(status))
	{
	}

public:
	WebSocketStatus status() const NOEXCEPT {
		return static_cast<WebSocketStatus>(code());
	}
};

}

#endif