// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2016, LH_Mouse. All wrongs reserved.

#include "../precompiled.hpp"
#include "client_writer.hpp"
#include "exception.hpp"
#include "urlencoded.hpp"
#include "../log.hpp"
#include "../profiler.hpp"
#include "../string.hpp"
#include "../buffer_streams.hpp"

namespace Poseidon {

namespace Http {
	ClientWriter::ClientWriter(){
	}
	ClientWriter::~ClientWriter(){
	}

	long ClientWriter::put_request(RequestHeaders request_headers, StreamBuffer entity){
		PROFILE_ME;

		StreamBuffer data;

		data.put(get_string_from_verb(request_headers.verb));
		data.put(' ');
		data.put(request_headers.uri);
		if(!request_headers.get_params.empty()){
			data.put('?');
			Buffer_ostream os;
			url_encode_params(os, request_headers.get_params);
			data.splice(os.get_buffer());
		}
		char temp[64];
		const unsigned ver_major = request_headers.version / 10000, ver_minor = request_headers.version % 10000;
		unsigned len = (unsigned)std::sprintf(temp, " HTTP/%u.%u\r\n", ver_major, ver_minor);
		data.put(temp, len);

		AUTO_REF(headers, request_headers.headers);
		if(entity.empty()){
			headers.erase("Content-Type");
			headers.erase("Transfer-Encoding");
			if((request_headers.verb == V_POST) || (request_headers.verb == V_PUT)){
				headers.set(sslit("Content-Length"), "0");
			} else {
				headers.erase("Content-Length");
			}
		} else {
			if(!headers.has("Content-Type")){
				headers.set(sslit("Content-Type"), "application/x-www-form-urlencoded");
			}
			headers.erase("Transfer-Encoding");
			len = (unsigned)std::sprintf(temp, "%llu", (unsigned long long)entity.size());
			headers.set(sslit("Content-Length"), std::string(temp, len));
		}

		for(AUTO(it, headers.begin()); it != headers.end(); ++it){
			data.put(it->first.get());
			data.put(": ");
			data.put(it->second);
			data.put("\r\n");
		}
		data.put("\r\n");

		data.splice(entity);

		return on_encoded_data_avail(STD_MOVE(data));
	}

	long ClientWriter::put_chunked_header(RequestHeaders request_headers){
		PROFILE_ME;

		StreamBuffer data;

		data.put(get_string_from_verb(request_headers.verb));
		data.put(' ');
		data.put(request_headers.uri);
		if(!request_headers.get_params.empty()){
			data.put('?');
			Buffer_ostream os;
			url_encode_params(os, request_headers.get_params);
			data.splice(os.get_buffer());
		}
		char temp[64];
		const unsigned ver_major = request_headers.version / 10000, ver_minor = request_headers.version % 10000;
		unsigned len = (unsigned)std::sprintf(temp, " HTTP/%u.%u\r\n", ver_major, ver_minor);
		data.put(temp, len);

		AUTO_REF(headers, request_headers.headers);
		if(!headers.has("Content-Type")){
			headers.set(sslit("Content-Type"), "application/x-www-form-urlencoded");
		}
		const AUTO_REF(transfer_encoding, headers.get("Transfer-Encoding"));
		if(transfer_encoding.empty() || (::strcasecmp(transfer_encoding.c_str(), "identity") == 0)){
			headers.set(sslit("Transfer-Encoding"), "chunked");
		}

		for(AUTO(it, headers.begin()); it != headers.end(); ++it){
			data.put(it->first.get());
			data.put(": ");
			data.put(it->second);
			data.put("\r\n");
		}
		data.put("\r\n");

		return on_encoded_data_avail(STD_MOVE(data));
	}
	long ClientWriter::put_chunk(StreamBuffer entity){
		PROFILE_ME;

		if(entity.empty()){
			LOG_POSEIDON_ERROR("You are not allowed to send an empty chunk");
			DEBUG_THROW(BasicException, sslit("You are not allowed to send an empty chunk"));
		}

		StreamBuffer chunk;

		char temp[64];
		unsigned len = (unsigned)std::sprintf(temp, "%llx\r\n", (unsigned long long)entity.size());
		chunk.put(temp, len);
		chunk.splice(entity);
		chunk.put("\r\n");

		return on_encoded_data_avail(STD_MOVE(chunk));
	}
	long ClientWriter::put_chunked_trailer(OptionalMap headers){
		PROFILE_ME;

		StreamBuffer data;

		data.put("0\r\n");
		for(AUTO(it, headers.begin()); it != headers.end(); ++it){
			data.put(it->first.get());
			data.put(": ");
			data.put(it->second);
			data.put("\r\n");
		}
		data.put("\r\n");

		return on_encoded_data_avail(STD_MOVE(data));
	}
}

}
