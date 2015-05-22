// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2015, LH_Mouse. All wrongs reserved.

#include "../precompiled.hpp"
#include "exception.hpp"
#include "../log.hpp"

namespace Poseidon {

namespace WebSocket {
	Exception::Exception(const char *file, std::size_t line, StatusCode statusCode, SharedNts message)
		: ProtocolException(file, line, STD_MOVE(message), static_cast<long>(statusCode))
	{
		LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_INFO,
			"WebSocket::Exception: statusCode = ", statusCode, ", what = ", what());
	}
	Exception::~Exception() NOEXCEPT {
	}
}

}
