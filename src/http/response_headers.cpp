// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2018, LH_Mouse. All wrongs reserved.

#include "../precompiled.hpp"
#include "response_headers.hpp"
#include "header_option.hpp"
#include "../buffer_streams.hpp"

namespace Poseidon {
namespace Http {

bool is_keep_alive_enabled(const Response_headers &response_headers) NOEXCEPT {
	const AUTO_REF(connection, response_headers.headers.get("Connection"));
	enum { opt_auto, opt_on, opt_off } opt = opt_auto;
	Buffer_istream is;
	is.set_buffer(Stream_buffer(connection));
	Header_option connection_option(is);
	if(is){
		if(::strcasecmp(connection_option.get_base().c_str(), "Keep-Alive") == 0){
			opt = opt_on;
		} else if(::strcasecmp(connection_option.get_base().c_str(), "Close") == 0){
			opt = opt_off;
		}
	}
	if(opt == opt_auto){
		if(response_headers.version < 10001){
			opt = opt_off;
		} else {
			opt = opt_on;
		}
	}
	return opt == opt_on;
}

std::pair<Response_headers, Stream_buffer> make_default_response(Status_code status_code, Optional_map headers){
	Response_headers response_headers;
	response_headers.version = 10001;
	response_headers.status_code = status_code;
	const AUTO(desc, get_status_code_desc(status_code));
	response_headers.reason = desc.desc_short;
	response_headers.headers = STD_MOVE(headers);

	Stream_buffer entity;
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
