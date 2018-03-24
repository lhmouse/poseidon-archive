// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2018, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_HTTP_REQUEST_HEADERS_HPP_
#define POSEIDON_HTTP_REQUEST_HEADERS_HPP_

#include "../cxx_ver.hpp"
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

extern bool is_keep_alive_enabled(const RequestHeaders &request_headers);

enum ContentEncoding {
	content_encoding_identity        =  0,
	content_encoding_deflate         =  1,
	content_encoding_gzip            =  2,
	content_encoding_not_acceptable  = 15,
};

extern ContentEncoding pick_content_encoding(const RequestHeaders &request_headers);

}
}

#endif
