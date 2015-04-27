// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2015, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_HTTP_RESPONSE_HEADERS_HPP_
#define POSEIDON_HTTP_RESPONSE_HEADERS_HPP_

#include "../cxx_ver.hpp"
#include <string>
#include "status_codes.hpp"
#include "../optional_map.hpp"

namespace Poseidon {

namespace Http {
	struct ResponseHeaders {
		unsigned version; // x * 10000 + y 表示 HTTP x.y
		StatusCode statusCode;
		std::string reason;
		OptionalMap headers;
	};

	inline void swap(ResponseHeaders &lhs, ResponseHeaders &rhs) NOEXCEPT {
		using std::swap;
		swap(lhs.version, rhs.version);
		swap(lhs.statusCode, rhs.statusCode);
		swap(lhs.reason, rhs.reason);
		swap(lhs.headers, rhs.headers);
	}
}

}

#endif
