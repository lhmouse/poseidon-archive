#ifndef POSEIDON_HTTP_EXCEPTION_HPP_
#define POSEIDON_HTTP_EXCEPTION_HPP_

#include "../exception.hpp"
#include "../optional_map.hpp"
#include "status.hpp"

namespace Poseidon {

class HttpException : public ProtocolException {
private:
	const OptionalMap m_headers;

public:
	HttpException(const char *file, std::size_t line,
		HttpStatus status, OptionalMap headers = OptionalMap());
	~HttpException() NOEXCEPT;

public:
	HttpStatus status() const NOEXCEPT {
		return static_cast<HttpStatus>(ProtocolException::code());
	}
	const OptionalMap &headers() const NOEXCEPT {
		return m_headers;
	}
};

}

#endif
