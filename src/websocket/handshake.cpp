// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2016, LH_Mouse. All wrongs reserved.

#include "../precompiled.hpp"
#include "handshake.hpp"
#include "../log.hpp"
#include "../sha1.hpp"
#include "../random.hpp"
#include "../profiler.hpp"
#include "../base64.hpp"

namespace Poseidon {

namespace WebSocket {
	Http::ResponseHeaders make_handshake_response(const Http::RequestHeaders &request){
		PROFILE_ME;

		Http::ResponseHeaders response = { };
		response.version = 10001;
		{
			if(request.version < 10001){
				LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_DEBUG, "HTTP 1.1 is required to use WebSocket");
				response.status_code = Http::ST_VERSION_NOT_SUPPORTED;
				goto _done;
			}
			if(request.verb != Http::V_GET){
				LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_DEBUG, "Must use GET verb to use WebSocket");
				response.status_code = Http::ST_METHOD_NOT_ALLOWED;
				goto _done;
			}
			const AUTO_REF(websocket_version, request.headers.get("Sec-WebSocket-Version"));
			char *endptr;
			const AUTO(version_num, std::strtol(websocket_version.c_str(), &endptr, 10));
			if(*endptr != 0){
				LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_DEBUG, "Unrecognized HTTP header Sec-WebSocket-Version: ", websocket_version);
				response.status_code = Http::ST_BAD_REQUEST;
				goto _done;
			}
			if((version_num < 0) || (version_num < 13)){
				LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_DEBUG, "Unsupported Sec-WebSocket-Version: ", websocket_version);
				response.status_code = Http::ST_BAD_REQUEST;
				goto _done;
			}
			const AUTO_REF(sec_websocket_key, request.headers.get("Sec-WebSocket-Key"));
			if(sec_websocket_key.empty()){
				LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_DEBUG, "No Sec-WebSocket-Key specified.");
				response.status_code = Http::ST_BAD_REQUEST;
				goto _done;
			}
			const AUTO(sha1, sha1_hash(sec_websocket_key + "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"));
			AUTO(sec_websocket_accept, base64_encode(sha1.data(), sha1.size()));
			response.headers.set(sslit("Upgrade"), "websocket");
			response.headers.set(sslit("Connection"), "Upgrade");
			response.headers.set(sslit("Sec-WebSocket-Accept"), STD_MOVE(sec_websocket_accept));
			response.status_code = Http::ST_SWITCHING_PROTOCOLS;
		}
	_done:
		response.reason = Http::get_status_code_desc(response.status_code).desc_short;
		return response;
	}

	std::pair<Http::RequestHeaders, std::string> make_handshake_request(std::string uri, OptionalMap get_params, std::string host){
		PROFILE_ME;

		Http::RequestHeaders request = { };
		request.verb       = Http::V_GET;
		request.uri        = STD_MOVE(uri);
		request.version    = 10001;
		request.get_params = STD_MOVE(get_params);
		request.headers.set(sslit("Host"), STD_MOVE(host));
		request.headers.set(sslit("Upgrade"), "websocket");
		request.headers.set(sslit("Connection"), "Keep-Alive");
		request.headers.set(sslit("Sec-WebSocket-Version"), "13");
		unsigned char random_bytes[24];
		for(unsigned i = 0; i < sizeof(random_bytes); ++i){
			random_bytes[i] = random_uint32();
		}
		std::string sec_websocket_key = base64_encode(random_bytes, sizeof(random_bytes));
		request.headers.set(sslit("Sec-WebSocket-Key"), sec_websocket_key);
		return std::make_pair(STD_MOVE_IDN(request), STD_MOVE_IDN(sec_websocket_key));
	}
	bool check_handshake_response(const Http::ResponseHeaders &response, const std::string &sec_websocket_key){
		PROFILE_ME;

		if(response.version < 10001){
			LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_DEBUG, "HTTP 1.1 is required to use WebSocket");
			return false;
		}
		if(response.status_code != Http::ST_SWITCHING_PROTOCOLS){
			LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_DEBUG, "Bad HTTP status code: ", response.status_code);
			return false;
		}
		const AUTO_REF(upgrade, response.headers.get("Upgrade"));
		if(upgrade.empty() || (::strcasecmp(upgrade.c_str(), "websocket") != 0)){
			LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_DEBUG, "Invalid Upgrade header: ", upgrade);
			return false;
		}
		const AUTO_REF(connection, response.headers.get("Connection"));
		if(connection.empty() || (::strcasecmp(connection.c_str(), "Upgrade") != 0)){
			LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_DEBUG, "Invalid Connection header: ", connection);
			return false;
		}
		const AUTO_REF(sec_websocket_accept, response.headers.get("Sec-WebSocket-Accept"));
		if(sec_websocket_accept.empty()){
			LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_DEBUG, "No Sec-WebSocket-Accept specified.");
			return false;
		}
		const AUTO(sha1, sha1_hash(sec_websocket_key + "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"));
		const AUTO(sec_websocket_accept_expecting, base64_encode(sha1.data(), sha1.size()));
		if(sec_websocket_accept != sec_websocket_accept_expecting){
			LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_DEBUG,
				"Bad Sec-WebSocket-Accept: got ", sec_websocket_accept, ", expecting ", sec_websocket_accept_expecting);
			return false;
		}
		return true;
	}
}

}
