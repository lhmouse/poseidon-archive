// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2018, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_SYSTEM_EXCEPTION_HPP_
#define POSEIDON_SYSTEM_EXCEPTION_HPP_

#include "exception.hpp"
#include <cerrno>

namespace Poseidon {

class System_exception : public Exception {
private:
	int m_code;

public:
	System_exception(const char *file, std::size_t line, const char *func, int code = errno);
	~System_exception() NOEXCEPT;

public:
	int get_code() const NOEXCEPT {
		return m_code;
	}
};

}

#endif
