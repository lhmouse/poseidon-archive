#ifndef POSEIDON_EXCEPTION_HPP_
#define POSEIDON_EXCEPTION_HPP_

#include "../cxx_ver.hpp"
#include <string>
#include <stdexcept>
#include <cstdio>
#include <cstddef>
#include <cstring>
#include <boost/cstdint.hpp>
#include <string.h>
#include <errno.h>
#include "utilities.hpp"

namespace Poseidon {

class Exception : public std::runtime_error {
protected:
	const char *const m_file;
	const std::size_t m_line;

public:
	Exception(const char *file, std::size_t line, std::string what)
		: std::runtime_error(STD_MOVE(what)), m_file(file), m_line(line)
	{
	}
	~Exception() throw() {
	}

public:
	const char *file() const throw() {
		return m_file;
	}
	std::size_t line() const throw() {
		return m_line;
	}
};

class SystemError : public Exception {
private:
	const int m_code;

public:
	SystemError(const char *file, std::size_t line, int code = errno)
		: Exception(file, line, getErrorDescAsString(code)), m_code(code)
	{
	}

public:
	int code() const throw() {
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
	ProtocolException(const char *file, std::size_t line, std::string what, int code)
		: Exception(file, line, STD_MOVE(what)), m_code(code)
	{
	}

public:
	int code() const throw() {
		return m_code;
	}
};

}

#define DEBUG_THROW(etype_, ...)	\
	throw etype_(__FILE__, __LINE__, __VA_ARGS__)

#endif
