#include "../../precompiled.hpp"
#include "exception.hpp"
using namespace Poseidon;

WebSocketException::WebSocketException(const char *file, std::size_t line,
	WebSocketStatus status, SharedNtmbs message)
	: ProtocolException(file, line, STD_MOVE(message), static_cast<unsigned>(status))
{
}
WebSocketException::~WebSocketException() NOEXCEPT {
}
