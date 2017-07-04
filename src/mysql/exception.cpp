// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2017, LH_Mouse. All wrongs reserved.

#include "../precompiled.hpp"
#include "exception.hpp"
#include "../log.hpp"

namespace Poseidon {

namespace MySql {
	Exception::Exception(const char *file, std::size_t line, const char *func, SharedNts schema, long code, SharedNts message)
		: ProtocolException(file, line, func, STD_MOVE(message), code)
		, m_schema(STD_MOVE(schema))
	{
		LOG_POSEIDON_ERROR("MySql::Exception: schema = ", get_schema(), ", code = ", get_code(), ", what = ", what());
	}
	Exception::~Exception() NOEXCEPT { }
}

}
