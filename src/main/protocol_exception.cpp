// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2015, LH_Mouse. All wrongs reserved.

#include "precompiled.hpp"
#include "protocol_exception.hpp"
#include "log.hpp"

namespace Poseidon {

ProtocolException::ProtocolException(const char *file, std::size_t line, SharedNts message, long code)
	: Exception(file, line, STD_MOVE(message)), m_code(code)
{
	LOG_POSEIDON_ERROR("Constructing ProtocolException: message = ", message, ", code = ", code);
}
ProtocolException::~ProtocolException() NOEXCEPT {
}

}
