// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2018, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_HTTP_RESPONSE_HEADERS_HPP_
#define POSEIDON_HTTP_RESPONSE_HEADERS_HPP_

#include "../cxx_ver.hpp"
#include "../fwd.hpp"
#include <string>
#include "status_codes.hpp"
#include "../optional_map.hpp"

namespace Poseidon {
namespace Http {

struct ResponseHeaders {
	unsigned version; // x * 10000 + y 表示 HTTP x.y
	StatusCode status_code;
	std::string reason;
	OptionalMap headers;
};

inline void swap(ResponseHeaders &lhs, ResponseHeaders &rhs) NOEXCEPT {
	using std::swap;
	swap(lhs.version, rhs.version);
	swap(lhs.status_code, rhs.status_code);
	swap(lhs.reason, rhs.reason);
	swap(lhs.headers, rhs.headers);
}

extern bool is_keep_alive_enabled(const ResponseHeaders &response_headers) NOEXCEPT;

extern std::pair<ResponseHeaders, StreamBuffer> make_default_response(StatusCode status_code, OptionalMap headers);

}
}

#endif
