// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2015, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_HTTP_CLIENT_READER_HPP_
#define POSEIDON_HTTP_CLIENT_READER_HPP_

#include <vector>
#include <string>
#include <cstddef>
#include <boost/cstdint.hpp>
#include "../stream_buffer.hpp"
#include "../optional_map.hpp"
#include "response_headers.hpp"

namespace Poseidon {

namespace Http {
	class ClientReader {
	private:
		enum State {
			S_FIRST_HEADER		= 0,
			S_HEADERS			= 1,
			S_IDENTITY			= 2,
			S_CHUNK_HEADER		= 3,
			S_CHUNK_DATA		= 4,
			S_CHUNKED_TRAILER	= 5,
		};

	protected:
		enum {
			CONTENT_CHUNKED		= (boost::uint64_t)-1,
			CONTENT_TILL_EOF	= (boost::uint64_t)-2,

			CONTENT_LENGTH_MAX	= (boost::uint64_t)-100,
		};

	private:
		enum {
			EXPECTING_NEW_LINE	= (boost::uint64_t)-99,
		};

	private:
		StreamBuffer m_queue;

		boost::uint64_t m_sizeExpecting;
		State m_state;

		ResponseHeaders m_responseHeaders;
		boost::uint64_t m_contentLength;
		boost::uint64_t m_contentOffset;

		boost::uint64_t m_chunkSize;
		boost::uint64_t m_chunkOffset;
		OptionalMap m_chunkedTrailer;

	protected:
		ClientReader();
		virtual ~ClientReader();

	protected:
		// transferEncoding 确保已被转换为小写并已排序。
		// 如果 Transfer-Encoding 为空或者不是 identity， contentLength 的值为 CONTENT_CHUNKED。
		virtual void onResponseHeaders(ResponseHeaders responseHeaders, std::string transferEncoding, boost::uint64_t contentLength) = 0;
		// 报文可能分几次收到。
		virtual void onResponseEntity(boost::uint64_t entityOffset, StreamBuffer entity) = 0;
		// 报文接收完毕。
		// 如果 onResponseHeaders() 的 contentLength 参数为 CONTENT_TILL_EOF，此处 realContentLength 即为实际接收大小。
		// 如果 onResponseHeaders() 的 contentLength 参数为 CONTENT_CHUNKED，使用这个函数标识结束。
		// chunked 允许追加报头。
		virtual bool onResponseEnd(boost::uint64_t contentLength, bool isChunked, OptionalMap headers) = 0;

	public:
		const StreamBuffer &getQueue() const {
			return m_queue;
		}
		StreamBuffer &getQueue(){
			return m_queue;
		}

		bool putEncodedData(StreamBuffer encoded);

		bool isContentTillEof() const;
		// 要求 isContentTillEof() 为 true，否则抛异常。
		bool terminateContent();
	};
}

}

#endif
