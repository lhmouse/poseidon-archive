// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2016, LH_Mouse. All wrongs reserved.

#include "precompiled.hpp"
#include "zlib.hpp"
#include "profiler.hpp"
#include "protocol_exception.hpp"
#include <zlib.h>

namespace Poseidon {

StreamBuffer gzip_deflate(StreamBuffer &src, int level){
	PROFILE_ME;

	StreamBuffer dst;
	::z_stream zs = { };
	try {
		int err_code = ::deflateInit2(&zs, level, Z_DEFLATED, 15 + 16, 8, Z_DEFAULT_STRATEGY);
		if(err_code != Z_OK){
			LOG_POSEIDON_ERROR("::deflateInit2() error: err_code = ", err_code);
			DEBUG_THROW(ProtocolException, sslit("::deflateInit2()"), err_code);
		}

		unsigned char out[4096], in[4096];
		zs.next_out = out;
		zs.avail_out = sizeof(out);
		zs.next_in = in;
		zs.avail_in = 0;
		for(;;){
			if(zs.avail_in == 0){
				const std::size_t avail = src.get(in, sizeof(in));
				if(avail == 0){
					break;
				}
				zs.next_in = in;
				zs.avail_in = avail;
			}
			err_code = ::deflate(&zs, Z_NO_FLUSH);
			if(err_code == Z_STREAM_END){
				break;
			}
			if(err_code != Z_OK){
				LOG_POSEIDON_ERROR("::deflate() error: err_code = ", err_code);
				DEBUG_THROW(ProtocolException, sslit("::deflate()"), err_code);
			}
			if(zs.avail_out == 0){
				dst.put(out, sizeof(out));
				zs.next_out = out;
				zs.avail_out = sizeof(out);
			}
		}
		if(zs.avail_out < sizeof(out)){
			dst.put(out, sizeof(out) - zs.avail_out);
		}

		::deflateEnd(&zs);
	} catch(...){
		::deflateEnd(&zs);
		throw;
	}
	return dst;
}
StreamBuffer gzip_inflate(StreamBuffer &src){
	PROFILE_ME;

	StreamBuffer dst;
	::z_stream zs = { };
	try {
		int err_code = ::inflateInit2(&zs, 15 + 16);
		if(err_code != Z_OK){
			LOG_POSEIDON_ERROR("::inflateInit2() error: err_code = ", err_code);
			DEBUG_THROW(ProtocolException, sslit("::inflateInit2()"), err_code);
		}

		unsigned char out[4096], in[4096];
		zs.next_out = out;
		zs.avail_out = sizeof(out);
		zs.next_in = in;
		zs.avail_in = 0;
		for(;;){
			if(zs.avail_in == 0){
				const std::size_t avail = src.get(in, sizeof(in));
				if(avail == 0){
					break;
				}
				zs.next_in = in;
				zs.avail_in = avail;
			}
			err_code = ::inflate(&zs, Z_NO_FLUSH);
			if(err_code == Z_STREAM_END){
				break;
			}
			if(err_code != Z_OK){
				LOG_POSEIDON_ERROR("::inflate() error: err_code = ", err_code);
				DEBUG_THROW(ProtocolException, sslit("::inflate()"), err_code);
			}
			if(zs.avail_out == 0){
				dst.put(out, sizeof(out));
				zs.next_out = out;
				zs.avail_out = sizeof(out);
			}
		}
		if(zs.avail_out < sizeof(out)){
			dst.put(out, sizeof(out) - zs.avail_out);
		}

		::inflateEnd(&zs);
	} catch(...){
		::inflateEnd(&zs);
		throw;
	}
	return dst;
}

}
