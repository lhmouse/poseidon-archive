// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2015, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_HTTP_SERVER_WRITER_HPP_
#define POSEIDON_HTTP_SERVER_WRITER_HPP_

#include <string>
#include <cstddef>
#include <boost/cstdint.hpp>
#include "../stream_buffer.hpp"
#include "../optional_map.hpp"
#include "response_headers.hpp"

namespace Poseidon {

namespace Http {
	class ServerWriter {
	public:
		ServerWriter();
		virtual ~ServerWriter();

	protected:
		virtual long onEncodedDataAvail(StreamBuffer encoded) = 0;

	public:
		long putResponseHeaders(ResponseHeaders responseHeaders);
		long putEntity(StreamBuffer data);

		long putResponse(ResponseHeaders responseHeaders, StreamBuffer entity);
		long putDefaultResponse(ResponseHeaders responseHeaders);

		long putChunkedHeader(ResponseHeaders responseHeaders);
		long putChunk(StreamBuffer entity);
		long putChunkedTrailer(OptionalMap headers);
	};
}

}

#endif
