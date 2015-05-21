// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2015, LH_Mouse. All wrongs reserved.

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

	long ClientWriter::putRequest(RequestHeaders requestHeaders, StreamBuffer entity){
		PROFILE_ME;

		StreamBuffer data;

		data.put(getStringFromVerb(requestHeaders.verb));
		data.put(' ');
		data.put(requestHeaders.uri);
		if(!requestHeaders.getParams.empty()){
			data.put('?');
			data.put(urlEncodedFromOptionalMap(requestHeaders.getParams));
		}
		char temp[64];
		const unsigned verMajor = requestHeaders.version / 10000, verMinor = requestHeaders.version % 10000;
		unsigned len = (unsigned)std::sprintf(temp, " HTTP/%u.%u\r\n", verMajor, verMinor);
		data.put(temp, len);

		AUTO_REF(headers, requestHeaders.headers);
		if(entity.empty()){
			headers.erase("Content-Type");
			headers.erase("Transfer-Encoding");
			headers.set("Content-Length", STR_0);
		} else {
			if(!headers.has("Content-Type")){
				headers.set("Content-Type", "application/x-www-form-urlencoded; charset=utf-8");
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

	long ClientWriter::putChunkedHeader(RequestHeaders requestHeaders){
		PROFILE_ME;

		StreamBuffer data;

		data.put(getStringFromVerb(requestHeaders.verb));
		data.put(' ');
		data.put(requestHeaders.uri);
		if(!requestHeaders.getParams.empty()){
			data.put('?');
			data.put(urlEncodedFromOptionalMap(requestHeaders.getParams));
		}
		char temp[64];
		const unsigned verMajor = requestHeaders.version / 10000, verMinor = requestHeaders.version % 10000;
		unsigned len = (unsigned)std::sprintf(temp, " HTTP/%u.%u\r\n", verMajor, verMinor);
		data.put(temp, len);

		AUTO_REF(headers, requestHeaders.headers);
		if(!headers.has("Content-Type")){
			headers.set("Content-Type", "application/x-www-form-urlencoded; charset=utf-8");
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
	long ClientWriter::putChunk(StreamBuffer entity){
		PROFILE_ME;

		StreamBuffer chunk;

		char temp[64];
		unsigned len = (unsigned)std::sprintf(temp, "%llx\r\n", (unsigned long long)entity.size());
		chunk.put(temp, len);
		chunk.splice(entity);
		chunk.put("\r\n");

		return onEncodedDataAvail(STD_MOVE(chunk));
	}
	long ClientWriter::putChunkEnd(OptionalMap headers){
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
