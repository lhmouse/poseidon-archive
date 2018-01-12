// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2018, LH_Mouse. All wrongs reserved.

#include "../precompiled.hpp"
#include "exception.hpp"
#include "../log.hpp"

namespace Poseidon {
namespace Cbpp {

Exception::Exception(const char *file, std::size_t line, const char *func, StatusCode status_code, SharedNts message)
	: BasicException(file, line, func, message)
	, m_status_code(status_code)
{
	LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_INFO, "Cbpp::Exception: status_code = ", get_status_code(), ", what = ", what());
}
Exception::~Exception() NOEXCEPT { }

}
}
