// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2016, LH_Mouse. All wrongs reserved.

#include "../precompiled.hpp"
#include "request_headers.hpp"
#include "header_option.hpp"
#include "../buffer_streams.hpp"
#include <string.h>

namespace Poseidon {

namespace Http {
	bool is_keep_alive_enabled(const RequestHeaders &request_headers) NOEXCEPT {
		enum { OPT_AUTO, OPT_ON, OPT_OFF } opt = OPT_AUTO;
		Buffer_istream is(StreamBuffer(request_headers.headers.get("Connection")));
		HeaderOption connection(is);
		if(is){
			if(::strcasecmp(connection.get_base().c_str(), "Keep-Alive") == 0){
				opt = OPT_ON;
			} else if(::strcasecmp(connection.get_base().c_str(), "Close") == 0){
				opt = OPT_OFF;
			}
		}
		if(opt == OPT_AUTO){
			if(request_headers.version < 10001){
				opt = OPT_OFF;
			} else {
				opt = OPT_ON;
			}
		}
		return opt == OPT_ON;
	}
}

}
