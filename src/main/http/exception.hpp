#ifndef POSEIDON_HTTP_EXCEPTION_HPP_
#define POSEIDON_HTTP_EXCEPTION_HPP_

#include "../exception.hpp"
#include "status.hpp"

namespace Poseidon {

class HttpException : public ProtocolException {
public:
	HttpException(const char *file, std::size_t line, HttpStatus status)
		: ProtocolException(file, line,
			getHttpStatusCodeDesc(status).descShort, static_cast<int>(status))
	{
	}

public:
	HttpStatus status() const NOEXCEPT {
		return static_cast<HttpStatus>(code());
	}
};

}

#endif
