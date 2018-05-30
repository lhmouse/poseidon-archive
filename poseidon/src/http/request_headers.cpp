// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2018, LH_Mouse. All wrongs reserved.

#include "../precompiled.hpp"
#include "request_headers.hpp"
#include "header_option.hpp"
#include "../buffer_streams.hpp"

namespace Poseidon {
namespace Http {

bool is_keep_alive_enabled(const Request_headers &request_headers){
	const AUTO_REF(connection, request_headers.headers.get("Connection"));
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
		if(request_headers.version < 10001){
			opt = opt_off;
		} else {
			opt = opt_on;
		}
	}
	return opt == opt_on;
}

Content_encoding pick_content_encoding(const Request_headers &request_headers){
	const AUTO_REF(accept_encoding, request_headers.headers.get("Accept-Encoding"));
	if(accept_encoding.empty()){
		return content_encoding_identity;
	}
	std::array<double, content_encoding_not_acceptable + 1> encodings;
	encodings.fill(-42);
	std::size_t begin = 0, end;
	Buffer_istream is;
	for(;;){
		end = accept_encoding.find(',', begin);
		if(end == std::string::npos){
			end = accept_encoding.size();
		}
		if(begin != end){
			is.clear();
			is.get_buffer().put(accept_encoding.data() + begin, end - begin);
			Header_option opt(is);
			const AUTO_REF(q_str, opt.get_option("q"));
			const double q = q_str.empty() ? 1.0 : std::strtod(q_str.c_str(), NULLPTR);
			if(!std::isnan(q) && (q >= 0)){
				if(::strcasecmp(opt.get_base().c_str(), "identity") == 0){
					encodings.at(content_encoding_identity) = q;
				} else if(::strcasecmp(opt.get_base().c_str(), "deflate") == 0){
					encodings.at(content_encoding_deflate) = q;
				} else if(::strcasecmp(opt.get_base().c_str(), "gzip") == 0){
					encodings.at(content_encoding_gzip) = q;
				} else if(::strcasecmp(opt.get_base().c_str(), "*") == 0){
					for(std::size_t i = 0; i < encodings.size(); ++i){
						if(encodings[i] < 0){
							encodings[i] = q;
						}
					}
				}
			}
		}
		if(end >= accept_encoding.size()){
			break;
		}
		begin = end + 1;
	}
	double identity_q = encodings.at(content_encoding_identity);
	if(identity_q < 0){
		identity_q = encodings.at(content_encoding_not_acceptable);
	}
	if(identity_q < 0){
		identity_q = 0.000001;
	}
	if(encodings.at(content_encoding_gzip) > identity_q){
		return content_encoding_gzip;
	} else if(encodings.at(content_encoding_deflate) > identity_q){
		return content_encoding_deflate;
	} else if(identity_q > 0){
		return content_encoding_identity;
	}
	return content_encoding_not_acceptable;
}

}
}
