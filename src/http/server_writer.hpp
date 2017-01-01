// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2017, LH_Mouse. All wrongs reserved.

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
		virtual long on_encoded_data_avail(StreamBuffer encoded) = 0;

	public:
		long put_response(ResponseHeaders response_headers, StreamBuffer entity);
		long put_default_response(ResponseHeaders response_headers);

		long put_chunked_header(ResponseHeaders response_headers);
		long put_chunk(StreamBuffer entity);
		long put_chunked_trailer(OptionalMap headers);
	};
}

}

#endif
