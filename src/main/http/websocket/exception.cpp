// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2015, LH_Mouse. All wrongs reserved.

#include "../../precompiled.hpp"
#include "exception.hpp"
#include "../../log.hpp"
using namespace Poseidon;

WebSocketException::WebSocketException(const char *file, std::size_t line,
	WebSocketStatus status, SharedNts message)
	: ProtocolException(file, line, STD_MOVE(message), static_cast<unsigned>(status))
{
	LOG_POSEIDON_ERROR("WebSocketException: status = ", static_cast<unsigned>(status), ", what = ", what());
}
WebSocketException::~WebSocketException() NOEXCEPT {
}
