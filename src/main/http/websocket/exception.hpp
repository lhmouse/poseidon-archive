#ifndef POSEIDON_HTTP_WEBSOCKET_EXCEPTION_HPP_
#define POSEIDON_HTTP_WEBSOCKET_EXCEPTION_HPP_

#include "../../exception.hpp"
#include "status.hpp"

namespace Poseidon {

class WebSocketException : public ProtocolException {
public:
	WebSocketException(const char *file, std::size_t line, WebSocketStatus status,
		const char *reason = NULLPTR)
		: ProtocolException(file, line,
			reason ? reason : getWebSocketStatusDesc(status),
			static_cast<int>(status))
	{
	}

public:
	WebSocketStatus status() const NOEXCEPT {
		return static_cast<WebSocketStatus>(code());
	}
};

}

#endif