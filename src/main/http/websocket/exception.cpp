#include "../../precompiled.hpp"
#include "exception.hpp"
using namespace Poseidon;

WebSocketException::WebSocketException(const char *file, std::size_t line,
	WebSocketStatus status, std::string reason)
	: ProtocolException(file, line, STD_MOVE(reason), static_cast<unsigned>(status))
{
}
WebSocketException::~WebSocketException() NOEXCEPT {
}
