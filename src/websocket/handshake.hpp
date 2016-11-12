// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2016, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_WEBSOCKET_HANDSHAKE_HPP_
#define POSEIDON_WEBSOCKET_HANDSHAKE_HPP_

#include "../http/request_headers.hpp"
#include "../http/response_headers.hpp"

namespace Poseidon {

namespace WebSocket {
	extern Http::ResponseHeaders make_handshake_response(const Http::RequestHeaders &request);

	extern std::pair<Http::RequestHeaders, std::string> make_handshake_request(std::string uri, OptionalMap get_params, std::string host);
	extern bool check_handshake_response(const Http::ResponseHeaders &response, const std::string &sec_websocket_key);
}

}

#endif
