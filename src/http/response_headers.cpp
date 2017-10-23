// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2017, LH_Mouse. All wrongs reserved.

#include "../precompiled.hpp"
#include "response_headers.hpp"
#include "header_option.hpp"
#include "../buffer_streams.hpp"

namespace Poseidon {
namespace Http {

bool is_keep_alive_enabled(const ResponseHeaders &response_headers) NOEXCEPT {
	enum { OPT_AUTO, OPT_ON, OPT_OFF } opt = OPT_AUTO;
	Buffer_istream is(StreamBuffer(response_headers.headers.get("Connection")));
	HeaderOption connection(is);
	if(is){
		if(::strcasecmp(connection.get_base().c_str(), "Keep-Alive") == 0){
			opt = OPT_ON;
		} else if(::strcasecmp(connection.get_base().c_str(), "Close") == 0){
			opt = OPT_OFF;
		}
	}
	if(opt == OPT_AUTO){
		if(response_headers.version < 10001){
			opt = OPT_OFF;
		} else {
			opt = OPT_ON;
		}
	}
	return opt == OPT_ON;
}

std::pair<ResponseHeaders, StreamBuffer> make_default_response(StatusCode status_code, OptionalMap headers){
	ResponseHeaders response_headers;
	response_headers.version = 10001;
	response_headers.status_code = status_code;
	const AUTO(desc, get_status_code_desc(status_code));
	response_headers.reason = desc.desc_short;
	response_headers.headers = STD_MOVE(headers);

	StreamBuffer entity;
	if(status_code / 100 >= 4){
		entity.put("<html><head><title>");
		entity.put(desc.desc_short);
		entity.put("</title></head><body><h1>");
		entity.put(desc.desc_short);
		entity.put("</h1><hr /><p>");
		entity.put(desc.desc_long);
		entity.put("</p></body></html>");

		response_headers.headers.set(sslit("Content-Type"), "text/html");
	}
	response_headers.headers.erase("Transfer-Encoding");
	response_headers.headers.set(sslit("Content-Length"), boost::lexical_cast<std::string>(entity.size()));

	return std::make_pair(STD_MOVE(response_headers), STD_MOVE(entity));
}

}
}
