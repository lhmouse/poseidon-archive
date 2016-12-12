// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2016, LH_Mouse. All wrongs reserved.

#include "../precompiled.hpp"
#include "server_writer.hpp"
#include "exception.hpp"
#include "../log.hpp"
#include "../profiler.hpp"
#include "../string.hpp"

namespace Poseidon {

namespace Http {
	ServerWriter::ServerWriter(){
	}
	ServerWriter::~ServerWriter(){
	}

	long ServerWriter::put_response(ResponseHeaders response_headers, StreamBuffer entity){
		PROFILE_ME;

		StreamBuffer data;

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
			headers.set(sslit("Content-Length"), "0");
		} else {
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
	long ServerWriter::put_default_response(ResponseHeaders response_headers){
		PROFILE_ME;

		StreamBuffer entity;

		const AUTO(status_code, response_headers.status_code);
		if(status_code / 100 >= 4){
			AUTO_REF(headers, response_headers.headers);

			headers.set(sslit("Content-Type"), "text/html");
			entity.put("<html><head><title>");
			const AUTO(desc, get_status_code_desc(status_code));
			entity.put(desc.desc_short);
			entity.put("</title></head><body><h1>");
			entity.put(desc.desc_short);
			entity.put("</h1><hr /><p>");
			entity.put(desc.desc_long);
			entity.put("</p></body></html>");
		}

		return put_response(STD_MOVE(response_headers), STD_MOVE(entity));
	}

	long ServerWriter::put_chunked_header(ResponseHeaders response_headers){
		PROFILE_ME;

		StreamBuffer data;

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
	long ServerWriter::put_chunk(StreamBuffer entity){
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
	long ServerWriter::put_chunked_trailer(OptionalMap headers){
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
