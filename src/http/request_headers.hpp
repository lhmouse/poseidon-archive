// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2017, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_HTTP_REQUEST_HEADERS_HPP_
#define POSEIDON_HTTP_REQUEST_HEADERS_HPP_

#include "../cxx_ver.hpp"
#include <string>
#include "verbs.hpp"
#include "../optional_map.hpp"

namespace Poseidon {
namespace Http {

struct RequestHeaders {
	Verb verb;
	std::string uri;
	unsigned version; // x * 10000 + y 表示 HTTP x.y
	OptionalMap get_params;
	OptionalMap headers;
};

inline void swap(RequestHeaders &lhs, RequestHeaders &rhs) NOEXCEPT {
	using std::swap;
	swap(lhs.verb, rhs.verb);
	swap(lhs.uri, rhs.uri);
	swap(lhs.version, rhs.version);
	swap(lhs.get_params, rhs.get_params);
	swap(lhs.headers, rhs.headers);
}

extern bool is_keep_alive_enabled(const RequestHeaders &request_headers);

enum ContentEncoding {
	CE_IDENTITY        =  0,
	CE_DEFLATE         =  1,
	CE_GZIP            =  2,
	CE_NOT_ACCEPTABLE  = 15,
};

extern ContentEncoding pick_content_encoding(const RequestHeaders &request_headers);

}
}

#endif
