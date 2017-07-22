// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2017, LH_Mouse. All wrongs reserved.

#include "precompiled.hpp"
#include "zlib.hpp"
#include "profiler.hpp"
#include "protocol_exception.hpp"
#define ZLIB_CONST 1
#include <zlib.h>

namespace Poseidon {

struct Deflator::Context : NONCOPYABLE {
	::z_stream stream;
	::Bytef temp[4096];

	explicit Context(bool gzip, int level){
		stream.zalloc = NULLPTR;
		stream.zfree = NULLPTR;
		stream.opaque = NULLPTR;
		stream.next_in = NULLPTR;
		stream.avail_in = 0;
		const int err_code = ::deflateInit2(&stream, level, Z_DEFLATED, 15 + gzip * 16, 9, Z_DEFAULT_STRATEGY);
		if(err_code < 0){
			LOG_POSEIDON_ERROR("::deflateInit2() error: err_code = ", err_code);
			DEBUG_THROW(ProtocolException, sslit("::deflateInit2()"), err_code);
		}
	}
	~Context(){
		const int err_code = ::deflateEnd(&stream);
		if(err_code < 0){
			LOG_POSEIDON_WARNING("::deflateEnd() error: err_code = ", err_code);
		}
	}
};

Deflator::Deflator(bool gzip, int level)
	: m_ctx(new Context(gzip, level)), m_buffer()
{ }
Deflator::~Deflator(){ }

void Deflator::clear(){
	PROFILE_ME;

	const int err_code = ::deflateReset(&(m_ctx->stream));
	if(err_code < 0){
		LOG_POSEIDON_FATAL("::deflateReset() error: err_code = ", err_code);
		std::abort();
	}
	m_buffer.clear();
}
void Deflator::put(const void *data, std::size_t size){
	PROFILE_ME;

	const ::Bytef *const begin = static_cast<const ::Bytef *>(data);
	m_ctx->stream.next_in = begin;
	m_ctx->stream.avail_in = 0;
	int err_code;
	for(;;){
		if(m_ctx->stream.avail_in == 0){
			std::size_t remaining = static_cast<std::size_t>(begin + size - m_ctx->stream.next_in);
			if(remaining > UINT_MAX){
				remaining = UINT_MAX;
			}
			m_ctx->stream.avail_in = remaining;
		}
		if(m_ctx->stream.avail_in == 0){
			break;
		}
		m_ctx->stream.next_out = m_ctx->temp;
		m_ctx->stream.avail_out = sizeof(m_ctx->temp);
		err_code = ::deflate(&(m_ctx->stream), Z_NO_FLUSH);
		if(err_code < 0){
			LOG_POSEIDON_ERROR("::deflate() error: err_code = ", err_code);
			DEBUG_THROW(ProtocolException, sslit("::deflate()"), err_code);
		}
		m_buffer.put(m_ctx->temp, static_cast<unsigned>(m_ctx->stream.next_out - m_ctx->temp));
		DEBUG_THROW_ASSERT(err_code == 0);
	}
}
void Deflator::put(const StreamBuffer &buffer){
	PROFILE_ME;

	const void *data;
	std::size_t size;
	StreamBuffer::EnumerationCookie cookie;
	while(buffer.enumerate_chunk(&data, &size, cookie)){
		put(data, size);
	}
}
void Deflator::flush(){
	PROFILE_ME;

	m_ctx->stream.next_in = NULLPTR;
	m_ctx->stream.avail_in = 0;
	int err_code;
	for(;;){
		m_ctx->stream.next_out = m_ctx->temp;
		m_ctx->stream.avail_out = sizeof(m_ctx->temp);
		err_code = ::deflate(&(m_ctx->stream), Z_SYNC_FLUSH);
		if(err_code == Z_BUF_ERROR){
			break;
		}
		if(err_code < 0){
			LOG_POSEIDON_ERROR("::deflate() error: err_code = ", err_code);
			DEBUG_THROW(ProtocolException, sslit("::deflate()"), err_code);
		}
		m_buffer.put(m_ctx->temp, static_cast<unsigned>(m_ctx->stream.next_out - m_ctx->temp));
		DEBUG_THROW_ASSERT(err_code == 0);
	}
}
StreamBuffer Deflator::finalize(){
	PROFILE_ME;

	m_ctx->stream.next_in = NULLPTR;
	m_ctx->stream.avail_in = 0;
	int err_code;
	for(;;){
		m_ctx->stream.next_out = m_ctx->temp;
		m_ctx->stream.avail_out = sizeof(m_ctx->temp);
		err_code = ::deflate(&(m_ctx->stream), Z_FINISH);
		if(err_code < 0){
			LOG_POSEIDON_ERROR("::deflate() error: err_code = ", err_code);
			DEBUG_THROW(ProtocolException, sslit("::deflate()"), err_code);
		}
		m_buffer.put(m_ctx->temp, static_cast<unsigned>(m_ctx->stream.next_out - m_ctx->temp));
		if(err_code == Z_STREAM_END){
			break;
		}
		DEBUG_THROW_ASSERT(err_code == 0);
	}

	AUTO(ret, STD_MOVE_IDN(m_buffer));
	clear();
	return ret;
}

struct Inflator::Context : NONCOPYABLE {
	::z_stream stream;
	::Bytef temp[4096];

	explicit Context(bool gzip){
		stream.zalloc = NULLPTR;
		stream.zfree = NULLPTR;
		stream.opaque = NULLPTR;
		stream.next_in = NULLPTR;
		stream.avail_in = 0;
		const int err_code = ::inflateInit2(&stream, 15 + gzip * 16);
		if(err_code < 0){
			LOG_POSEIDON_ERROR("::inflateInit2() error: err_code = ", err_code);
			DEBUG_THROW(ProtocolException, sslit("::deflateInit2()"), err_code);
		}
	}
	~Context(){
		const int err_code = ::inflateEnd(&stream);
		if(err_code < 0){
			LOG_POSEIDON_WARNING("::inflateEnd() error: err_code = ", err_code);
		}
	}
};

Inflator::Inflator(bool gzip)
	: m_ctx(new Context(gzip)), m_buffer()
{ }
Inflator::~Inflator(){ }

void Inflator::clear(){
	PROFILE_ME;

	const int err_code = ::inflateReset(&(m_ctx->stream));
	if(err_code < 0){
		LOG_POSEIDON_FATAL("::inflateReset() error: err_code = ", err_code);
		std::abort();
	}
	m_buffer.clear();
}
void Inflator::put(const void *data, std::size_t size){
	PROFILE_ME;

	const ::Bytef *const begin = static_cast<const ::Bytef *>(data);
	m_ctx->stream.next_in = begin;
	m_ctx->stream.avail_in = 0;
	int err_code;
	for(;;){
		if(m_ctx->stream.avail_in == 0){
			std::size_t remaining = static_cast<std::size_t>(begin + size - m_ctx->stream.next_in);
			if(remaining > UINT_MAX){
				remaining = UINT_MAX;
			}
			m_ctx->stream.avail_in = remaining;
		}
		if(m_ctx->stream.avail_in == 0){
			break;
		}
		m_ctx->stream.next_out = m_ctx->temp;
		m_ctx->stream.avail_out = sizeof(m_ctx->temp);
		err_code = ::inflate(&(m_ctx->stream), Z_NO_FLUSH);
		if(err_code < 0){
			LOG_POSEIDON_ERROR("::inflate() error: err_code = ", err_code);
			DEBUG_THROW(ProtocolException, sslit("::inflate()"), err_code);
		}
		m_buffer.put(m_ctx->temp, static_cast<unsigned>(m_ctx->stream.next_out - m_ctx->temp));
		if(err_code == Z_STREAM_END){
			break;
		}
		DEBUG_THROW_ASSERT(err_code == 0);
	}
}
void Inflator::put(const StreamBuffer &buffer){
	PROFILE_ME;

	const void *data;
	std::size_t size;
	StreamBuffer::EnumerationCookie cookie;
	while(buffer.enumerate_chunk(&data, &size, cookie)){
		put(data, size);
	}
}
void Inflator::flush(){
	PROFILE_ME;

	m_ctx->stream.next_in = NULLPTR;
	m_ctx->stream.avail_in = 0;
	int err_code;
	for(;;){
		m_ctx->stream.next_out = m_ctx->temp;
		m_ctx->stream.avail_out = sizeof(m_ctx->temp);
		err_code = ::inflate(&(m_ctx->stream), Z_SYNC_FLUSH);
		if(err_code == Z_BUF_ERROR){
			break;
		}
		if(err_code < 0){
			LOG_POSEIDON_ERROR("::inflate() error: err_code = ", err_code);
			DEBUG_THROW(ProtocolException, sslit("::inflate()"), err_code);
		}
		m_buffer.put(m_ctx->temp, static_cast<unsigned>(m_ctx->stream.next_out - m_ctx->temp));
		DEBUG_THROW_ASSERT(err_code == 0);
	}
}
StreamBuffer Inflator::finalize(){
	PROFILE_ME;

	m_ctx->stream.next_in = NULLPTR;
	m_ctx->stream.avail_in = 0;
	int err_code;
	for(;;){
		m_ctx->stream.next_out = m_ctx->temp;
		m_ctx->stream.avail_out = sizeof(m_ctx->temp);
		err_code = ::inflate(&(m_ctx->stream), Z_FINISH);
		if(err_code < 0){
			LOG_POSEIDON_ERROR("::inflate() error: err_code = ", err_code);
			DEBUG_THROW(ProtocolException, sslit("::inflate()"), err_code);
		}
		m_buffer.put(m_ctx->temp, static_cast<unsigned>(m_ctx->stream.next_out - m_ctx->temp));
		if(err_code == Z_STREAM_END){
			break;
		}
		DEBUG_THROW_ASSERT(err_code == 0);
	}

	AUTO(ret, STD_MOVE_IDN(m_buffer));
	clear();
	return ret;
}

}
