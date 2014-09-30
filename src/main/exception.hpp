#ifndef POSEIDON_EXCEPTION_HPP_
#define POSEIDON_EXCEPTION_HPP_

#include "../cxx_ver.hpp"
#include <string>
#include <stdexcept>
#include <cerrno>
#include <cstddef>

namespace Poseidon {

class Exception : public std::runtime_error {
protected:
	const char *const m_file;
	const std::size_t m_line;

public:
	Exception(const char *file, std::size_t line, std::string what);

public:
	const char *file() const NOEXCEPT {
		return m_file;
	}
	std::size_t line() const NOEXCEPT {
		return m_line;
	}
};

class SystemError : public Exception {
private:
	const int m_code;

public:
	SystemError(const char *file, std::size_t line, int code = errno);

public:
	int code() const NOEXCEPT {
		return m_code;
	}
};

class ProtocolException : public Exception {
public:
	enum {
		ERR_END_OF_STREAM	= -1,
		ERR_BAD_REQUEST		= -2,
		ERR_INTERNAL_ERROR	= -3,
	};

private:
	const int m_code;

public:
	ProtocolException(const char *file, std::size_t line, std::string what, int code);

public:
	int code() const NOEXCEPT {
		return m_code;
	}
};

}

#define DEBUG_THROW(etype_, ...)	throw etype_(__FILE__, __LINE__, __VA_ARGS__)

#endif
