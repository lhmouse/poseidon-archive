// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2018, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_HTTP_RESPONSE_HEADERS_HPP_
#define POSEIDON_HTTP_RESPONSE_HEADERS_HPP_

#include "../cxx_ver.hpp"
#include "status_codes.hpp"
#include "../optional_map.hpp"
#include "../stream_buffer.hpp"

namespace Poseidon {
namespace Http {

struct Response_headers {
	unsigned version; // x * 10000 + y 表示 HTTP x.y
	Status_code status_code;
	std::string reason;
	Optional_map headers;
};

extern bool is_keep_alive_enabled(const Response_headers &response_headers) NOEXCEPT;

extern std::pair<Response_headers, Stream_buffer> make_default_response(Status_code status_code, Optional_map headers);

}
}

#endif
