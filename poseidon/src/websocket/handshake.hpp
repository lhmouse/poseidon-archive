// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2018, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_WEBSOCKET_HANDSHAKE_HPP_
#define POSEIDON_WEBSOCKET_HANDSHAKE_HPP_

#include "../http/request_headers.hpp"
#include "../http/response_headers.hpp"

namespace Poseidon {
namespace Websocket {

extern Http::Response_headers make_handshake_response(const Http::Request_headers &request);

extern std::pair<Http::Request_headers, std::string> make_handshake_request(std::string uri, Option_map get_params, std::string host);
extern bool check_handshake_response(const Http::Response_headers &response, const std::string &sec_websocket_key);

}
}

#endif
