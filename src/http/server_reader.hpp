// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2016, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_HTTP_SERVER_READER_HPP_
#define POSEIDON_HTTP_SERVER_READER_HPP_

#include <string>
#include <cstddef>
#include <boost/cstdint.hpp>
#include "../stream_buffer.hpp"
#include "../optional_map.hpp"
#include "request_headers.hpp"

namespace Poseidon {

namespace Http {
	class ServerReader {
	private:
		enum State {
			S_FIRST_HEADER      = 0,
			S_HEADERS           = 1,
			S_IDENTITY          = 2,
			S_CHUNK_HEADER      = 3,
			S_CHUNK_DATA        = 4,
			S_CHUNKED_TRAILER   = 5,
		};

	protected:
		enum {
			CONTENT_CHUNKED     = (boost::uint64_t)-1,

			CONTENT_LENGTH_MAX  = (boost::uint64_t)-100,
		};

	private:
		enum {
			EXPECTING_NEW_LINE  = (boost::uint64_t)-99,
		};

	private:
		StreamBuffer m_queue;

		boost::uint64_t m_size_expecting;
		State m_state;

		RequestHeaders m_request_headers;
		boost::uint64_t m_content_length;
		boost::uint64_t m_content_offset;

		boost::uint64_t m_chunk_size;
		boost::uint64_t m_chunk_offset;
		OptionalMap m_chunked_trailer;

	protected:
		ServerReader();
		virtual ~ServerReader();

	protected:
		// 如果 Transfer-Encoding 为 chunked， content_length 的值为 CONTENT_CHUNKED。
		virtual void on_request_headers(RequestHeaders request_headers, boost::uint64_t content_length) = 0;
		// 报文可能分几次收到。
		virtual void on_request_entity(boost::uint64_t entity_offset, StreamBuffer entity) = 0;
		// 报文接收完毕。
		// 如果 on_request_headers() 的 content_length 参数为 CONTENT_CHUNKED，使用这个函数标识结束。
		// chunked 允许追加报头。
		virtual bool on_request_end(boost::uint64_t content_length, OptionalMap headers) = 0;

	public:
		const StreamBuffer &get_queue() const {
			return m_queue;
		}
		StreamBuffer &get_queue(){
			return m_queue;
		}

		bool put_encoded_data(StreamBuffer encoded, bool dont_parse_get_params = false);
	};
}

}

#endif
