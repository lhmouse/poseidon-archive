// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2016, LH_Mouse. All wrongs reserved.

#include "../precompiled.hpp"
#include "exception.hpp"
#include "../log.hpp"

namespace Poseidon {

namespace MySql {
	Exception::Exception(const char *file, std::size_t line, long error_code, SharedNts message)
		: ProtocolException(file, line, STD_MOVE(message), error_code)
	{
		LOG_POSEIDON_ERROR("MySql::Exception: error_code = ", error_code, ", what = ", what());
	}
	Exception::~Exception() NOEXCEPT {
	}
}

}
