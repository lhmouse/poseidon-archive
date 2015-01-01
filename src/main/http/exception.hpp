// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2015, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_HTTP_EXCEPTION_HPP_
#define POSEIDON_HTTP_EXCEPTION_HPP_

#include "../exception.hpp"
#include "../optional_map.hpp"
#include "status.hpp"

namespace Poseidon {

class HttpException : public ProtocolException {
private:
	boost::shared_ptr<OptionalMap> m_headers;

public:
	HttpException(const char *file, std::size_t line,
		HttpStatus status, OptionalMap headers = OptionalMap());
	~HttpException() NOEXCEPT;

public:
	HttpStatus status() const NOEXCEPT {
		return static_cast<HttpStatus>(ProtocolException::code());
	}
	const OptionalMap &headers() const NOEXCEPT;
};

}

#endif
