#ifndef POSEIDON_EXCEPTION_HPP_
#define POSEIDON_EXCEPTION_HPP_

#include <string>
#include <exception>
#include <boost/cstdint.hpp>

namespace Poseidon {

class Exception : public std::exception {
protected:
	const std::string m_reason;

public:
	explicit Exception(const std::string &reason)
		: m_reason(reason);
	{
	}

public:
	const char *what() const throw() {
		return m_reason.c_str();
	}
};

class FatalError : public Exception {
public:
	explicit FatalError(const std::string &reason)
		: Exception(reason)
	{
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
	const boost::int32_t m_code;

public:
	ProtocolException(const std::string &reason, boost::int32_t code)
		: Exception(reason), m_code(code)
	{
	}

public:
	boost::uint32_t getErrorCode() const throw() {
		return m_code;
	}
};

}

#endif
