// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2017, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_PROTOCOL_EXCEPTION_HPP_
#define POSEIDON_PROTOCOL_EXCEPTION_HPP_

#include "exception.hpp"

namespace Poseidon {

class ProtocolException : public Exception {
private:
	long m_code;

public:
	ProtocolException(const char *file, std::size_t line, const char *func, SharedNts message, long code);
	~ProtocolException() NOEXCEPT;

public:
	long get_code() const NOEXCEPT {
		return m_code;
	}
};

}

#endif
