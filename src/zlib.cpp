// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2016, LH_Mouse. All wrongs reserved.

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

		const int err_code = ::deflateInit2(&stream, level, Z_DEFLATED, 15 + gzip * 16, 9, Z_DEFAULT_STRATEGY);
		if(err_code < 0){
			LOG_POSEIDON_ERROR("::deflateInit2() error: err_code = ", err_code);
			DEBUG_THROW(ProtocolException, sslit("::deflateInit2()"), err_code);
		}

		stream.next_out = temp;
		stream.avail_out = sizeof(temp);
	}
	~Context(){
		const int err_code = ::deflateEnd(&stream);
		if(err_code < 0){
			LOG_POSEIDON_ERROR("::deflateEnd() error: err_code = ", err_code);
			DEBUG_THROW(ProtocolException, sslit("::deflateEnd()"), err_code);
		}
	}
};

Deflator::Deflator(bool gzip, int level)
	: m_context(new Context(gzip, level)), m_buffer()
{
}
Deflator::~Deflator(){
}

void Deflator::clear(){
	PROFILE_ME;

	const int err_code = ::deflateReset(&(m_context->stream));
	if(err_code < 0){
		LOG_POSEIDON_ERROR("::deflateReset() error: err_code = ", err_code);
		DEBUG_THROW(ProtocolException, sslit("::deflateReset()"), err_code);
	}
	m_buffer.clear();
}
void Deflator::put(const void *data, std::size_t size){
	PROFILE_ME;

	m_context->stream.next_in = static_cast<const ::Bytef *>(data);
	m_context->stream.avail_in = size;
	while(m_context->stream.avail_in != 0){
		const int err_code = ::deflate(&(m_context->stream), Z_NO_FLUSH);
		if(err_code == Z_STREAM_END){
			break;
		}
		if(err_code < 0){
			LOG_POSEIDON_ERROR("::deflate() error: err_code = ", err_code);
			DEBUG_THROW(ProtocolException, sslit("::deflate()"), err_code);
		}
		if(m_context->stream.avail_out == 0){
			m_buffer.put(m_context->temp, sizeof(m_context->temp));
			m_context->stream.next_out = m_context->temp;
			m_context->stream.avail_out = sizeof(m_context->temp);
		}
	}
}
void Deflator::put(const StreamBuffer &buffer){
	PROFILE_ME;

	for(AUTO(en, buffer.get_chunk_enumerator()); en; ++en){
		put(en.data(), en.size());
	}
}
StreamBuffer Deflator::finalize(){
	PROFILE_ME;

	m_context->stream.next_in = m_context->temp;
	m_context->stream.avail_in = 0;
	for(;;){
		const int err_code = ::deflate(&(m_context->stream), Z_FINISH);
		if(err_code == Z_STREAM_END){
			break;
		}
		if(err_code < 0){
			LOG_POSEIDON_ERROR("::deflate() error: err_code = ", err_code);
			DEBUG_THROW(ProtocolException, sslit("::deflate()"), err_code);
		}
		if(m_context->stream.avail_out == 0){
			m_buffer.put(m_context->temp, sizeof(m_context->temp));
			m_context->stream.next_out = m_context->temp;
			m_context->stream.avail_out = sizeof(m_context->temp);
		}
	}
	m_buffer.put(m_context->temp, sizeof(m_context->temp) - m_context->stream.avail_out);

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

		const int err_code = ::inflateInit2(&stream, 15 + gzip * 16);
		if(err_code < 0){
			LOG_POSEIDON_ERROR("::inflateInit2() error: err_code = ", err_code);
			DEBUG_THROW(ProtocolException, sslit("::deflateInit2()"), err_code);
		}

		stream.next_out = temp;
		stream.avail_out = sizeof(temp);
	}
	~Context(){
		const int err_code = ::inflateEnd(&stream);
		if(err_code < 0){
			LOG_POSEIDON_ERROR("::inflateEnd() error: err_code = ", err_code);
			DEBUG_THROW(ProtocolException, sslit("::inflateEnd()"), err_code);
		}
	}
};

Inflator::Inflator(bool gzip)
	: m_context(new Context(gzip)), m_buffer()
{
}
Inflator::~Inflator(){
}

void Inflator::clear(){
	PROFILE_ME;

	const int err_code = ::inflateReset(&(m_context->stream));
	if(err_code < 0){
		LOG_POSEIDON_ERROR("::inflateReset() error: err_code = ", err_code);
		DEBUG_THROW(ProtocolException, sslit("::inflateReset()"), err_code);
	}
	m_buffer.clear();
}
void Inflator::put(const void *data, std::size_t size){
	PROFILE_ME;

	m_context->stream.next_in = static_cast<const ::Bytef *>(data);
	m_context->stream.avail_in = size;
	while(m_context->stream.avail_in != 0){
		const int err_code = ::inflate(&(m_context->stream), Z_NO_FLUSH);
		if(err_code == Z_STREAM_END){
			break;
		}
		if(err_code < 0){
			LOG_POSEIDON_ERROR("::inflate() error: err_code = ", err_code);
			DEBUG_THROW(ProtocolException, sslit("::inflate()"), err_code);
		}
		if(m_context->stream.avail_out == 0){
			m_buffer.put(m_context->temp, sizeof(m_context->temp));
			m_context->stream.next_out = m_context->temp;
			m_context->stream.avail_out = sizeof(m_context->temp);
		}
	}
}
void Inflator::put(const StreamBuffer &buffer){
	PROFILE_ME;

	for(AUTO(en, buffer.get_chunk_enumerator()); en; ++en){
		put(en.data(), en.size());
	}
}
StreamBuffer Inflator::finalize(){
	PROFILE_ME;

	m_context->stream.next_in = m_context->temp;
	m_context->stream.avail_in = 0;
	for(;;){
		const int err_code = ::inflate(&(m_context->stream), Z_FINISH);
		if(err_code == Z_STREAM_END){
			break;
		}
		if(err_code < 0){
			LOG_POSEIDON_ERROR("::inflate() error: err_code = ", err_code);
			DEBUG_THROW(ProtocolException, sslit("::inflate()"), err_code);
		}
		if(m_context->stream.avail_out == 0){
			m_buffer.put(m_context->temp, sizeof(m_context->temp));
			m_context->stream.next_out = m_context->temp;
			m_context->stream.avail_out = sizeof(m_context->temp);
		}
	}
	m_buffer.put(m_context->temp, sizeof(m_context->temp) - m_context->stream.avail_out);

	AUTO(ret, STD_MOVE_IDN(m_buffer));
	clear();
	return ret;
}

}
