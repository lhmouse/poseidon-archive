// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2015, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_CBPP_EXCEPTION_HPP_
#define POSEIDON_CBPP_EXCEPTION_HPP_

#include "../exception.hpp"
#include "status_codes.hpp"

namespace Poseidon {

namespace Cbpp {
	class MessageException : public ProtocolException {
	public:
		MessageException(const char *file, std::size_t line, StatusCode statusCode, SharedNts message = SharedNts());
		~MessageException() NOEXCEPT;

	public:
		StatusCode statusCode() const NOEXCEPT {
			return static_cast<StatusCode>(code());
		}
	};
}

}

#endif
