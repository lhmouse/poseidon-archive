// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2015, LH_Mouse. All wrongs reserved.

#include "../precompiled.hpp"
#include "server_writer.hpp"
#include "const_strings.hpp"
#include "exception.hpp"
#include "utilities.hpp"
#include "../log.hpp"
#include "../profiler.hpp"
#include "../string.hpp"

namespace Poseidon {

namespace Http {
	ServerWriter::ServerWriter(){
	}
	ServerWriter::~ServerWriter(){
	}

	long ServerWriter::putResponse(ResponseHeaders responseHeaders, StreamBuffer entity){
		PROFILE_ME;

		StreamBuffer data;

		const unsigned verMajor = responseHeaders.version / 10000, verMinor = responseHeaders.version % 10000;
		const unsigned statusCode = static_cast<unsigned>(responseHeaders.statusCode);
		char temp[64];
		unsigned len = (unsigned)std::sprintf(temp, "HTTP/%u.%u %u ", verMajor,verMinor, statusCode);
		data.put(temp, len);
		data.put(responseHeaders.reason);
		data.put("\r\n");

		AUTO_REF(headers, responseHeaders.headers);
		if(entity.empty()){
			headers.erase("Content-Type");
			headers.erase("Transfer-Encoding");
			headers.set("Content-Length", STR_0);
		} else {
			if(!headers.has("Content-Type")){
				headers.set("Content-Type", "text/plain; charset=utf-8");
			}

			AUTO(transferEncoding, headers.get("Transfer-Encoding"));
			AUTO(pos, transferEncoding.find(';'));
			if(pos != std::string::npos){
				transferEncoding.erase(pos);
			}
			transferEncoding = toLowerCase(trim(STD_MOVE(transferEncoding)));

			if(transferEncoding.empty() || (transferEncoding == STR_IDENTITY)){
				headers.set("Content-Length", boost::lexical_cast<std::string>(entity.size()));
			} else {
				headers.erase("Content-Length");

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

		return onEncodedDataAvail(STD_MOVE(data));
	}
	long ServerWriter::putDefaultResponse(ResponseHeaders responseHeaders){
		PROFILE_ME;

		StreamBuffer entity;

		const AUTO(statusCode, responseHeaders.statusCode);
		if(statusCode / 100 >= 4){
			AUTO_REF(headers, responseHeaders.headers);

			headers.set("Content-Type", "text/html; charset=utf-8");
			entity.put("<html><head><title>");
			const AUTO(desc, getStatusCodeDesc(statusCode));
			entity.put(desc.descShort);
			entity.put("</title></head><body><h1>");
			entity.put(desc.descShort);
			entity.put("</h1><hr /><p>");
			entity.put(desc.descLong);
			entity.put("</p></body></html>");
		}

		return putResponse(STD_MOVE(responseHeaders), STD_MOVE(entity));
	}

	long ServerWriter::putChunkedHeader(ResponseHeaders responseHeaders){
		PROFILE_ME;

		StreamBuffer data;

		const unsigned verMajor = responseHeaders.version / 10000, verMinor = responseHeaders.version % 10000;
		const unsigned statusCode = static_cast<unsigned>(responseHeaders.statusCode);
		char temp[64];
		unsigned len = (unsigned)std::sprintf(temp, "HTTP/%u.%u %u ", verMajor,verMinor, statusCode);
		data.put(temp, len);
		data.put(responseHeaders.reason);
		data.put("\r\n");

		AUTO_REF(headers, responseHeaders.headers);
		if(!headers.has("Content-Type")){
			headers.set("Content-Type", "text/plain; charset=utf-8");
		}

		AUTO(transferEncoding, headers.get("Transfer-Encoding"));
		AUTO(pos, transferEncoding.find(';'));
		if(pos != std::string::npos){
			transferEncoding.erase(pos);
		}
		transferEncoding = toLowerCase(trim(STD_MOVE(transferEncoding)));

		if(transferEncoding.empty() || (transferEncoding == STR_IDENTITY)){
			headers.set("Transfer-Encoding", STR_CHUNKED);
		} else {
			headers.set("Transfer-Encoding", STD_MOVE(transferEncoding));
		}
		headers.erase("Content-Length");

		for(AUTO(it, headers.begin()); it != headers.end(); ++it){
			data.put(it->first.get());
			data.put(": ");
			data.put(it->second);
			data.put("\r\n");
		}
		data.put("\r\n");

		return onEncodedDataAvail(STD_MOVE(data));
	}
	long ServerWriter::putChunk(StreamBuffer entity){
		PROFILE_ME;

		StreamBuffer chunk;

		char temp[64];
		unsigned len = (unsigned)std::sprintf(temp, "%llx\r\n", (unsigned long long)entity.size());
		chunk.put(temp, len);
		chunk.splice(entity);
		chunk.put("\r\n");

		return onEncodedDataAvail(STD_MOVE(chunk));
	}
	long ServerWriter::putChunkedTrailer(OptionalMap headers){
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

		return onEncodedDataAvail(STD_MOVE(data));
	}
}

}
