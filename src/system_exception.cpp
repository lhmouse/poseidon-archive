// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2018, LH_Mouse. All wrongs reserved.

#include "precompiled.hpp"
#include "system_exception.hpp"
#include "log.hpp"
#include "errno.hpp"

namespace Poseidon {

SystemException::SystemException(const char *file, std::size_t line, const char *func, int code)
	: Exception(file, line, func, get_error_desc(code)), m_code(code)
{
	LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_INFO, "Constructing SystemException: code = ", get_code(), ", what = ", what());
}
SystemException::~SystemException() NOEXCEPT {
	//
}

}
