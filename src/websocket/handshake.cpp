// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2015, LH_Mouse. All wrongs reserved.

#include "../precompiled.hpp"
#include "handshake.hpp"
#include "../log.hpp"
#include "../hash.hpp"
#include "../http/utilities.hpp"

namespace Poseidon {

namespace WebSocket {
	Http::ResponseHeaders make_handshake_response(const Http::RequestHeaders &request_headers){
		Http::ResponseHeaders ret;

		if(request_headers.version < 10001){
			LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_DEBUG, "HTTP 1.1 is required to use WebSocket");
			ret.version = 10000;
			ret.status_code = Http::ST_METHOD_NOT_ALLOWED;
			return ret;
		}
		ret.version = 10001;

		if(request_headers.verb != Http::V_GET){
			LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_DEBUG, "Must use GET verb to use WebSocket");
			ret.status_code = Http::ST_METHOD_NOT_ALLOWED;
			return ret;
		}

		AUTO_REF(websocket_version, request_headers.headers.get("Sec-WebSocket-Version"));
		if(websocket_version != "13"){
			LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_DEBUG, "Unknown HTTP header Sec-WebSocket-Version: ", websocket_version);
			ret.status_code = Http::ST_BAD_REQUEST;
			return ret;
		}

		AUTO(sec_web_socket_key, request_headers.headers.get("Sec-WebSocket-Key"));
		if(sec_web_socket_key.empty()){
			LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_DEBUG, "No Sec-WebSocket-Key specified.");
			ret.status_code = Http::ST_BAD_REQUEST;
			return ret;
		}
		sec_web_socket_key += "258EAFA5-E914-47DA-95CA-C5AB0DC85B11";
		const AUTO(sha1, sha1_hash(sec_web_socket_key));
		AUTO(sec_web_socket_accept, Http::base64_encode(sha1.data(), sha1.size()));
		ret.headers.set(sslit("Upgrade"), "websocket");
		ret.headers.set(sslit("Connection"), "Upgrade");
		ret.headers.set(sslit("Sec-WebSocket-Accept"), STD_MOVE(sec_web_socket_accept));
		ret.status_code = Http::ST_SWITCHING_PROTOCOLS;
		return ret;
	}
}

}
