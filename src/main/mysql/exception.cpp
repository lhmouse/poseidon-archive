// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2015, LH_Mouse. All wrongs reserved.

#include "../precompiled.hpp"
#include "exception.hpp"
#include "../log.hpp"
using namespace Poseidon;

MySqlException::MySqlException(const char *file, std::size_t line,
	unsigned code, SharedNts message)
	: ProtocolException(file, line, STD_MOVE(message), code)
{
	LOG_POSEIDON(Logger::SP_CRITICAL | Logger::LV_ERROR,
		"MySqlException: code = ", code, ", what = ", what());
}
MySqlException::~MySqlException() NOEXCEPT {
}
