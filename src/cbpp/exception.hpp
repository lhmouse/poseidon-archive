// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2015, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_CBPP_EXCEPTION_HPP_
#define POSEIDON_CBPP_EXCEPTION_HPP_

#include "../protocol_exception.hpp"
#include "status_codes.hpp"

namespace Poseidon {

namespace Cbpp {
	class Exception : public ProtocolException {
	public:
		Exception(const char *file, std::size_t line, StatusCode status_code, SharedNts message = SharedNts());
		~Exception() NOEXCEPT;

	public:
		StatusCode status_code() const NOEXCEPT {
			return static_cast<StatusCode>(code());
		}
	};
}

}

#endif
