#ifndef POSEIDON_EXCEPTION_HPP_
#define POSEIDON_EXCEPTION_HPP_

#include <string>
#include <exception>
#include <cstdio>
#include <cstddef>
#include <cstring>
#include <boost/cstdint.hpp>
#include <string.h>
#include <errno.h>
#include "utilities.hpp"

namespace Poseidon {

class Exception : public std::exception {
protected:
	const char *const m_file;
	const std::size_t m_line;
	const std::string m_what;

public:
	Exception(const char *file, std::size_t line, const std::string &what)
		: m_file(file), m_line(line), m_what(what)
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
	const char *what() const throw() {
		return m_what.c_str();
	}
};

class SystemError : public Exception {
private:
	const int m_code;

public:
	SystemError(const char *file, std::size_t line, int code = errno)
		: Exception(file, line, getErrorDesc(code).get()), m_code(code)
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
		ERR_BROKEN_HEADER	= -1,
		ERR_TRUNCATED_BODY	= -2,
		ERR_END_OF_STREAM	= -3,
		ERR_BAD_REQUEST		= -4,
		ERR_INTERNAL_ERROR	= -5,
	};

private:
	const int m_code;

public:
	ProtocolException(const char *file, std::size_t line, const std::string &what, int code)
		: Exception(file, line, what), m_code(code)
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
