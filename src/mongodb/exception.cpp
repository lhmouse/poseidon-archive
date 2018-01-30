// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2018, LH_Mouse. All wrongs reserved.

#include "../precompiled.hpp"
#include "exception.hpp"
#include "../log.hpp"

namespace Poseidon {
namespace MongoDb {

Exception::Exception(const char *file, std::size_t line, const char *func, SharedNts database, unsigned long code, SharedNts message)
	: BasicException(file, line, func, STD_MOVE(message))
	, m_database(STD_MOVE(database)), m_code(code)
{
	LOG_POSEIDON_ERROR("MongoDb::Exception: database = ", get_database(), ", code = ", get_code(), ", what = ", what());
}
Exception::~Exception() NOEXCEPT {
	//
}

}
}
