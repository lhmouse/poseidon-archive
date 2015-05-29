// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2015, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_HTTP_CLIENT_WRITER_HPP_
#define POSEIDON_HTTP_CLIENT_WRITER_HPP_

#include <string>
#include <cstddef>
#include <boost/cstdint.hpp>
#include "../stream_buffer.hpp"
#include "../optional_map.hpp"
#include "request_headers.hpp"

namespace Poseidon {

namespace Http {
	class ClientWriter {
	public:
		ClientWriter();
		virtual ~ClientWriter();

	protected:
		virtual long onEncodedDataAvail(StreamBuffer encoded) = 0;

	public:
		long putRequestHeaders(RequestHeaders requestHeaders);
		long putEntity(StreamBuffer data);

		long putRequest(RequestHeaders requestHeaders, StreamBuffer entity = StreamBuffer());

		long putChunkedHeader(RequestHeaders requestHeaders);
		long putChunk(StreamBuffer entity);
		long putChunkedTrailer(OptionalMap headers);
	};
}

}

#endif
