// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2018, LH_Mouse. All wrongs reserved.

#include "../precompiled.hpp"
#include "server_writer.hpp"
#include "exception.hpp"
#include "../log.hpp"
#include "../profiler.hpp"
#include "../string.hpp"

namespace Poseidon {
namespace Http {

Server_writer::Server_writer(){
	//
}
Server_writer::~Server_writer(){
	//
}

long Server_writer::put_response(Response_headers response_headers, Stream_buffer entity, bool set_content_length){
	PROFILE_ME;

	Stream_buffer data;

	const unsigned ver_major = response_headers.version / 10000, ver_minor = response_headers.version % 10000;
	const unsigned status_code = static_cast<unsigned>(response_headers.status_code);
	char temp[64];
	unsigned len = (unsigned)std::sprintf(temp, "HTTP/%u.%u %u ", ver_major, ver_minor, status_code);
	data.put(temp, len);
	data.put(response_headers.reason);
	data.put("\r\n");

	AUTO_REF(headers, response_headers.headers);
	if(entity.empty()){
		headers.erase("Content-Type");
		headers.erase("Transfer-Encoding");
		if(set_content_length){
			headers.set(Rcnts::view("Content-Length"), "0");
		}
	} else {
		headers.erase("Transfer-Encoding");
		if(set_content_length){
			len = (unsigned)std::sprintf(temp, "%llu", (unsigned long long)entity.size());
			headers.set(Rcnts::view("Content-Length"), std::string(temp, len));
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

long Server_writer::put_chunked_header(Response_headers response_headers){
	PROFILE_ME;

	Stream_buffer data;

	const unsigned ver_major = response_headers.version / 10000, ver_minor = response_headers.version % 10000;
	const unsigned status_code = static_cast<unsigned>(response_headers.status_code);
	char temp[64];
	unsigned len = (unsigned)std::sprintf(temp, "HTTP/%u.%u %u ", ver_major, ver_minor, status_code);
	data.put(temp, len);
	data.put(response_headers.reason);
	data.put("\r\n");

	AUTO_REF(headers, response_headers.headers);
	const AUTO_REF(transfer_encoding, headers.get("Transfer-Encoding"));
	if(transfer_encoding.empty() || (::strcasecmp(transfer_encoding.c_str(), "identity") == 0)){
		headers.set(Rcnts::view("Transfer-Encoding"), "chunked");
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
long Server_writer::put_chunk(Stream_buffer entity){
	PROFILE_ME;
	DEBUG_THROW_UNLESS(!entity.empty(), Basic_exception, Rcnts::view("You are not allowed to send an empty chunk"));

	Stream_buffer chunk;

	char temp[64];
	unsigned len = (unsigned)std::sprintf(temp, "%llx\r\n", (unsigned long long)entity.size());
	chunk.put(temp, len);
	chunk.splice(entity);
	chunk.put("\r\n");

	return on_encoded_data_avail(STD_MOVE(chunk));
}
long Server_writer::put_chunked_trailer(Optional_map headers){
	PROFILE_ME;

	Stream_buffer data;

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
