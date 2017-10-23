// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2017, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_HTTP_EXCEPTION_HPP_
#define POSEIDON_HTTP_EXCEPTION_HPP_

#include "../protocol_exception.hpp"
#include "../optional_map.hpp"
#include "status_codes.hpp"

namespace Poseidon {
namespace Http {

class Exception : public ProtocolException {
private:
	boost::shared_ptr<OptionalMap> m_headers;

public:
	Exception(const char *file, std::size_t line, const char *func, StatusCode status_code, OptionalMap headers = OptionalMap());
	~Exception() NOEXCEPT;

public:
	StatusCode get_status_code() const NOEXCEPT {
		return static_cast<StatusCode>(get_code());
	}
	const OptionalMap &get_headers() const NOEXCEPT;
};

}
}

#endif
