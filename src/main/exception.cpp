// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2015, LH_Mouse. All wrongs reserved.

#include "precompiled.hpp"
#include "exception.hpp"
#include "log.hpp"
#include "utilities.hpp"
using namespace Poseidon;

Exception::Exception(const char *file, std::size_t line, SharedNts message)
	: m_file(file), m_line(line), m_message(STD_MOVE(message))
{
	LOG_POSEIDON_ERROR("Constructing Exception: file = ", m_file,
		", line = ", m_line, ", what = ", m_message);
}
Exception::~Exception() NOEXCEPT {
}

SystemError::SystemError(const char *file, std::size_t line, int code)
	: Exception(file, line, getErrorDesc(code)), m_code(code)
{
}
SystemError::~SystemError() NOEXCEPT {
}

ProtocolException::ProtocolException(const char *file, std::size_t line, SharedNts message, long code)
	: Exception(file, line, STD_MOVE(message)), m_code(code)
{
}
ProtocolException::~ProtocolException() NOEXCEPT {
}
