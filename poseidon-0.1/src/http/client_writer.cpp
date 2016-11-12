// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2016, LH_Mouse. All wrongs reserved.

#include "../precompiled.hpp"
#include "client_writer.hpp"
#include "const_strings.hpp"
#include "exception.hpp"
#include "utilities.hpp"
#include "../log.hpp"
#include "../profiler.hpp"
#include "../string.hpp"

namespace Poseidon {

namespace Http {
	ClientWriter::ClientWriter(){
	}
	ClientWriter::~ClientWriter(){
	}

	long ClientWriter::put_request_headers(RequestHeaders request_headers){
		PROFILE_ME;

		StreamBuffer data;

		data.put(get_string_from_verb(request_headers.verb));
		data.put(' ');
		data.put(request_headers.uri);
		if(!request_headers.get_params.empty()){
			data.put('?');
			data.put(url_encoded_from_optional_map(request_headers.get_params));
		}
		char temp[64];
		const unsigned ver_major = request_headers.version / 10000, ver_minor = request_headers.version % 10000;
		unsigned len = (unsigned)std::sprintf(temp, " HTTP/%u.%u\r\n", ver_major, ver_minor);
		data.put(temp, len);

		AUTO_REF(headers, request_headers.headers);
		for(AUTO(it, headers.begin()); it != headers.end(); ++it){
			data.put(it->first.get());
			data.put(": ");
			data.put(it->second);
			data.put("\r\n");
		}
		data.put("\r\n");

		return on_encoded_data_avail(STD_MOVE(data));
	}

	long ClientWriter::put_request(RequestHeaders request_headers, StreamBuffer entity){
		PROFILE_ME;

		StreamBuffer data;

		data.put(get_string_from_verb(request_headers.verb));
		data.put(' ');
		data.put(request_headers.uri);
		if(!request_headers.get_params.empty()){
			data.put('?');
			data.put(url_encoded_from_optional_map(request_headers.get_params));
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
				headers.set(sslit("Content-Length"), STR_0);
			} else {
				headers.erase("Content-Length");
			}
		} else {
			if(!headers.has("Content-Type")){
				headers.set(sslit("Content-Type"), "application/x-www-form-urlencoded; charset=utf-8");
			}

			AUTO(transfer_encoding, headers.get("Transfer-Encoding"));
			AUTO(pos, transfer_encoding.find(';'));
			if(pos != std::string::npos){
				transfer_encoding.erase(pos);
			}
			transfer_encoding = to_lower_case(trim(STD_MOVE(transfer_encoding)));

			if(transfer_encoding.empty() || (transfer_encoding == STR_IDENTITY)){
				headers.set(sslit("Content-Length"), boost::lexical_cast<std::string>(entity.size()));
			} else {
				// 只有一个 chunk。
				StreamBuffer chunk;
				len = (unsigned)std::sprintf(temp, "%llx\r\n", (unsigned long long)entity.size());
				chunk.put(temp, len);
				chunk.splice(entity);
				chunk.put("\r\n0\r\n\r\n");
				entity.swap(chunk);
			}
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
	long ClientWriter::put_entity(StreamBuffer data){
		PROFILE_ME;

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
			data.put(url_encoded_from_optional_map(request_headers.get_params));
		}
		char temp[64];
		const unsigned ver_major = request_headers.version / 10000, ver_minor = request_headers.version % 10000;
		unsigned len = (unsigned)std::sprintf(temp, " HTTP/%u.%u\r\n", ver_major, ver_minor);
		data.put(temp, len);

		AUTO_REF(headers, request_headers.headers);
		if(!headers.has("Content-Type")){
			headers.set(sslit("Content-Type"), "application/x-www-form-urlencoded; charset=utf-8");
		}

		AUTO(transfer_encoding, headers.get("Transfer-Encoding"));
		AUTO(pos, transfer_encoding.find(';'));
		if(pos != std::string::npos){
			transfer_encoding.erase(pos);
		}
		transfer_encoding = to_lower_case(trim(STD_MOVE(transfer_encoding)));

		if(transfer_encoding.empty() || (transfer_encoding == STR_IDENTITY)){
			headers.set(sslit("Transfer-Encoding"), STR_CHUNKED);
		} else {
			headers.set(sslit("Transfer-Encoding"), STD_MOVE(transfer_encoding));
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
