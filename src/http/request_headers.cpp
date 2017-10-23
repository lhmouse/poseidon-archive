// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2017, LH_Mouse. All wrongs reserved.

#include "../precompiled.hpp"
#include "request_headers.hpp"
#include "header_option.hpp"
#include "../buffer_streams.hpp"

namespace Poseidon {
namespace Http {

bool is_keep_alive_enabled(const RequestHeaders &request_headers){
	const AUTO_REF(connection, request_headers.headers.get("Connection"));
	enum { R_AUTO, R_ON, R_OFF } result = R_AUTO;
	Buffer_istream is;
	is.get_buffer().put(connection);
	HeaderOption opt(is);
	if(is){
		if(::strcasecmp(opt.get_base().c_str(), "Keep-Alive") == 0){
			result = R_ON;
		} else if(::strcasecmp(opt.get_base().c_str(), "Close") == 0){
			result = R_OFF;
		}
	}
	if(result == R_AUTO){
		if(request_headers.version < 10001){
			result = R_OFF;
		} else {
			result = R_ON;
		}
	}
	return result == R_ON;
}

ContentEncoding pick_content_encoding(const RequestHeaders &request_headers){
	const AUTO_REF(accept_encoding, request_headers.headers.get("Accept-Encoding"));
	if(accept_encoding.empty()){
		return CE_IDENTITY;
	}
	boost::array<double, CE_NOT_ACCEPTABLE + 1> encodings;
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
			HeaderOption opt(is);
			const AUTO_REF(q_str, opt.get_option("q"));
			const double q = q_str.empty() ? 1.0 : std::strtod(q_str.c_str(), NULLPTR);
			if(!std::isnan(q) && (q >= 0)){
				if(::strcasecmp(opt.get_base().c_str(), "identity") == 0){
					encodings.at(CE_IDENTITY) = q;
				} else if(::strcasecmp(opt.get_base().c_str(), "deflate") == 0){
					encodings.at(CE_DEFLATE) = q;
				} else if(::strcasecmp(opt.get_base().c_str(), "gzip") == 0){
					encodings.at(CE_GZIP) = q;
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
	double identity_q = encodings.at(CE_IDENTITY);
	if(identity_q < 0){
		identity_q = encodings.at(CE_NOT_ACCEPTABLE);
	}
	if(identity_q < 0){
		identity_q = 0.000001;
	}
	if(encodings.at(CE_GZIP) > identity_q){
		return CE_GZIP;
	} else if(encodings.at(CE_DEFLATE) > identity_q){
		return CE_DEFLATE;
	} else if(identity_q > 0){
		return CE_IDENTITY;
	}
	return CE_NOT_ACCEPTABLE;
}

}
}
