// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2015, LH_Mouse. All wrongs reserved.

#include "../precompiled.hpp"
#include "exception.hpp"
#include "../log.hpp"

namespace Poseidon {

namespace MySql {
	SqlException::SqlException(const char *file, std::size_t line, unsigned code, SharedNts message)
		: ProtocolException(file, line, STD_MOVE(message), code)
	{
		LOG_POSEIDON_ERROR("SqlException: code = ", code, ", what = ", what());
	}
	SqlException::~SqlException() NOEXCEPT {
	}
}

}
