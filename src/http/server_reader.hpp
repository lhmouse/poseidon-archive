// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2015, LH_Mouse. All wrongs reserved.

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

		RequestHeaders m_requestHeaders;
		boost::uint64_t m_contentLength;
		boost::uint64_t m_contentOffset;

		boost::uint64_t m_chunkSize;
		boost::uint64_t m_chunkOffset;
		OptionalMap m_chunkedTrailer;

	protected:
		ServerReader();
		virtual ~ServerReader();

	protected:
		// transferEncoding 确保已被转换为小写并已排序。
		// 如果 Transfer-Encoding 为空或者不是 identity， contentLength 的值为 CONTENT_CHUNKED。
		virtual void onRequestHeaders(RequestHeaders requestHeaders, std::string transferEncoding, boost::uint64_t contentLength) = 0;
		// 报文可能分几次收到。
		virtual void onRequestEntity(boost::uint64_t entityOffset, StreamBuffer entity) = 0;
		// 报文接收完毕。
		// 如果 onRequestHeaders() 的 contentLength 参数为 CONTENT_CHUNKED，使用这个函数标识结束。
		// chunked 允许追加报头。
		virtual bool onRequestEnd(boost::uint64_t contentLength, bool isChunked, OptionalMap headers) = 0;

	public:
		const StreamBuffer &getQueue() const {
			return m_queue;
		}
		StreamBuffer &getQueue(){
			return m_queue;
		}

		bool putEncodedData(StreamBuffer encoded);
	};
}

}

#endif
