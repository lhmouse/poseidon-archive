// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2015, LH_Mouse. All wrongs reserved.

#include "../precompiled.hpp"
#include "handshake.hpp"
#include "../log.hpp"
#include "../hash.hpp"
#include "../http/utilities.hpp"

namespace Poseidon {

namespace WebSocket {
	Http::ResponseHeaders makeHandshakeResponse(const Http::RequestHeaders &requestHeaders){
		Http::ResponseHeaders ret;

		if(requestHeaders.version < 10001){
			LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_DEBUG, "HTTP 1.1 is required to use WebSocket");
			ret.version = 10000;
			ret.statusCode = Http::ST_METHOD_NOT_ALLOWED;
			return ret;
		}
		ret.version = 10001;

		if(requestHeaders.verb != Http::V_GET){
			LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_DEBUG, "Must use GET verb to use WebSocket");
			ret.statusCode = Http::ST_METHOD_NOT_ALLOWED;
			return ret;
		}

		AUTO_REF(websocketVersion, requestHeaders.headers.get("Sec-WebSocket-Version"));
		if(websocketVersion != "13"){
			LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_DEBUG, "Unknown HTTP header Sec-WebSocket-Version: ", websocketVersion);
			ret.statusCode = Http::ST_BAD_REQUEST;
			return ret;
		}

		AUTO(secWebSocketKey, requestHeaders.headers.get("Sec-WebSocket-Key"));
		if(secWebSocketKey.empty()){
			LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_DEBUG, "No Sec-WebSocket-Key specified.");
			ret.statusCode = Http::ST_BAD_REQUEST;
			return ret;
		}
		secWebSocketKey += "258EAFA5-E914-47DA-95CA-C5AB0DC85B11";
		const AUTO(sha1, sha1Hash(secWebSocketKey));
		AUTO(secWebSocketAccept, Http::base64Encode(sha1.data(), sha1.size()));
		ret.headers.set("Upgrade", "websocket");
		ret.headers.set("Connection", "Upgrade");
		ret.headers.set("Sec-WebSocket-Accept", STD_MOVE(secWebSocketAccept));
		ret.statusCode = Http::ST_SWITCHING_PROTOCOLS;
		return ret;
	}
}

}
