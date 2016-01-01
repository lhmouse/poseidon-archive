// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2016, LH_Mouse. All wrongs reserved.

#include "precompiled.hpp"
#include "exception.hpp"
#include "log.hpp"

namespace Poseidon {

Exception::Exception(const char *fi, std::size_t ln, SharedNts msg)
	: m_file(fi), m_line(ln), m_message(STD_MOVE(msg))
{
	LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_INFO,
		"Constructing Exception: file = ", m_file, ", line = ", m_line, ", what = ", m_message);
}
Exception::~Exception() NOEXCEPT {
}

}
