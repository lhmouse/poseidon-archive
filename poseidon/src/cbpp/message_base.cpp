// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2018, LH_Mouse. All wrongs reserved.

#include "../precompiled.hpp"
#include "message_base.hpp"
#include "exception.hpp"
#include "status_codes.hpp"
#include "../log.hpp"
#include "../vint64.hpp"

namespace Poseidon {
namespace Cbpp {

Message_base::~Message_base(){
	//
}

void shift_vint(std::int64_t &value, Stream_buffer &buf, const char *name){
	POSEIDON_LOG_TRACE("Shifting out `vint`: ", name);
	Stream_buffer::Read_iterator rit(buf);
	if(!vint64_from_binary(value, rit, SIZE_MAX)){
		POSEIDON_THROW(Exception, status_end_of_stream, Rcnts::view("End of stream encountered"));
	}
}
void shift_vuint(std::uint64_t &value, Stream_buffer &buf, const char *name){
	POSEIDON_LOG_TRACE("Shifting out `vuint`: ", name);
	Stream_buffer::Read_iterator rit(buf);
	if(!vuint64_from_binary(value, rit, SIZE_MAX)){
		POSEIDON_THROW(Exception, status_end_of_stream, Rcnts::view("End of stream encountered"));
	}
}
void shift_string(std::string &value, Stream_buffer &buf, const char *name){
	POSEIDON_LOG_TRACE("Shifting out `string`: ", name);
	std::uint64_t length;
	Stream_buffer::Read_iterator rit(buf);
	if(!vuint64_from_binary(length, rit, SIZE_MAX)){
		POSEIDON_THROW(Exception, status_end_of_stream, Rcnts::view("End of stream encountered"));
	}
	if(length > PTRDIFF_MAX){
		POSEIDON_THROW(Exception, status_length_error, Rcnts::view("String length too large"));
	}
	value.resize(static_cast<std::size_t>(length));
	if((length != 0) && (buf.get(&value[0], value.size()) != length)){
		POSEIDON_THROW(Exception, status_end_of_stream, Rcnts::view("End of stream encountered"));
	}
}
void shift_blob(Stream_buffer &value, Stream_buffer &buf, const char *name){
	POSEIDON_LOG_TRACE("Shifting out `blob`: ", name);
	std::uint64_t length;
	Stream_buffer::Read_iterator rit(buf);
	if(!vuint64_from_binary(length, rit, SIZE_MAX)){
		POSEIDON_THROW(Exception, status_end_of_stream, Rcnts::view("End of stream encountered"));
	}
	if(length > PTRDIFF_MAX){
		POSEIDON_THROW(Exception, status_length_error, Rcnts::view("String length too large"));
	}
	value = buf.cut_off(static_cast<std::size_t>(length));
	if(value.size() != length){
		POSEIDON_THROW(Exception, status_end_of_stream, Rcnts::view("End of stream encountered"));
	}
}
void shift_fixed(void *data, std::size_t size, Stream_buffer &buf, const char *name){
	POSEIDON_LOG_TRACE("Shifting out `fixed`: ", name);
	if(buf.get(data, size) != size){
		POSEIDON_THROW(Exception, status_end_of_stream, Rcnts::view("End of stream encountered"));
	}
}
void shift_flexible(Stream_buffer &value, Stream_buffer &buf, const char *name){
	POSEIDON_LOG_TRACE("Shifting out `flexible`: ", name);
	value.clear();
	value.swap(buf);
}

void push_vint(Stream_buffer &buf, std::int64_t value){
	Stream_buffer::Write_iterator wit(buf);
	vint64_to_binary(value, wit);
}
void push_vuint(Stream_buffer &buf, std::uint64_t value){
	Stream_buffer::Write_iterator wit(buf);
	vuint64_to_binary(value, wit);
}
void push_string(Stream_buffer &buf, const std::string &value){
	Stream_buffer::Write_iterator wit(buf);
	vuint64_to_binary(value.size(), wit);
	buf.put(value);
}
void push_blob(Stream_buffer &buf, const Stream_buffer &value){
	Stream_buffer::Write_iterator wit(buf);
	vuint64_to_binary(value.size(), wit);
	buf.put(value);
}
void push_fixed(Stream_buffer &buf, const void *data, std::size_t size){
	buf.put(data, size);
}
void push_flexible(Stream_buffer &buf, const Stream_buffer &value){
	buf.put(value);
}

}
}
