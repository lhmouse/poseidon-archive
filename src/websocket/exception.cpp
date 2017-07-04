// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2017, LH_Mouse. All wrongs reserved.

#include "../precompiled.hpp"
#include "exception.hpp"
#include "../log.hpp"

namespace Poseidon {

namespace WebSocket {
	Exception::Exception(const char *file, std::size_t line, const char *func, StatusCode status_code, SharedNts message)
		: ProtocolException(file, line, func, STD_MOVE(message), static_cast<long>(status_code))
	{
		LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_INFO,
			"WebSocket::Exception: code = ", get_code(), ", what = ", what());
	}
	Exception::~Exception() NOEXCEPT { }
}

}
