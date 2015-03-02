// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2015, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_EXCEPTION_HPP_
#define POSEIDON_EXCEPTION_HPP_

#include "cxx_ver.hpp"
#include <exception>
#include <cerrno>
#include <cstddef>
#include "shared_nts.hpp"

namespace Poseidon {

class Exception : public std::exception {
protected:
	const char *m_file;
	std::size_t m_line;
	SharedNts m_message; // 拷贝构造函数不抛出异常。

public:
	Exception(const char *file, std::size_t line, SharedNts message);
	~Exception() NOEXCEPT;

public:
	const char *what() const NOEXCEPT {
		return m_message.get();
	}

	const char *file() const NOEXCEPT {
		return m_file;
	}
	std::size_t line() const NOEXCEPT {
		return m_line;
	}
};

class SystemError : public Exception {
private:
	int m_code;

public:
	SystemError(const char *file, std::size_t line, int code = errno);
	~SystemError() NOEXCEPT;

public:
	int code() const NOEXCEPT {
		return m_code;
	}
};

class ProtocolException : public Exception {
private:
	unsigned m_code;

public:
	ProtocolException(const char *file, std::size_t line, SharedNts message, long code);
	~ProtocolException() NOEXCEPT;

public:
	long code() const NOEXCEPT {
		return m_code;
	}
};

}

#define DEBUG_THROW(etype_, ...)	\
	do {	\
		etype_ e_(__FILE__, __LINE__, ## __VA_ARGS__);	\
		throw e_;	\
	} while(false)

#endif
