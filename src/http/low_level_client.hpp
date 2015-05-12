// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2015, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_HTTP_LOW_LEVEL_CLIENT_HPP_
#define POSEIDON_HTTP_LOW_LEVEL_CLIENT_HPP_

#include "../cxx_ver.hpp"
#include <string>
#include <cstddef>
#include <boost/shared_ptr.hpp>
#include <boost/cstdint.hpp>
#include "request_headers.hpp"
#include "response_headers.hpp"
#include "../optional_map.hpp"
#include "../tcp_client_base.hpp"
#include "../stream_buffer.hpp"

namespace Poseidon {

namespace Http {
	class LowLevelClient : public TcpClientBase {
	private:
		class ResponseHeaderJob;
		class EntityJob;
		class ContentEofJob;
		class ChunkedTrailerJob;

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

			CONTENT_LENGTH_MAX	= (boost::uint64_t)-256,
		};

	private:
		StreamBuffer m_received;

		bool m_expectingNewLine;
		boost::uint64_t m_sizeExpecting;
		State m_state;

		ResponseHeaders m_responseHeaders;
		boost::uint64_t m_contentLength;
		boost::uint64_t m_contentOffset;

		boost::uint64_t m_chunkSize;
		boost::uint64_t m_chunkOffset;
		OptionalMap m_chunkTrailer;

	protected:
		LowLevelClient(const SockAddr &addr, bool useSsl);
		LowLevelClient(const IpPort &addr, bool useSsl);
		~LowLevelClient();

	protected:
		void onReadAvail(const void *data, std::size_t size) OVERRIDE;

		void onReadHup() NOEXCEPT OVERRIDE;

		// transferEncoding 确保已被转换为小写、已排序，并且 chunked 和 identity 被移除（如果有的话）。
		// 如果 Transfer-Encoding 为空或者不是 identity， contentLength 的值为 CONTENT_CHUNKED。
		// 如果没有指定 Content-Length 同时也不是 chunked，contentLength 的值为 CONTENT_TILL_EOF。
		virtual void onLowLevelResponseHeaders(ResponseHeaders responseHeaders,
			std::vector<std::string> transferEncoding, boost::uint64_t contentLength) = 0;
		// 报文可能分几次收到。
		virtual void onLowLevelEntity(boost::uint64_t contentOffset, StreamBuffer entity) = 0;
		// 如果 onResponseHeaders() 的 contentLength 参数为 CONTENT_CHUNKED，使用这个函数标识结束。
		// chunked 允许追加报头。
		virtual void onLowLevelChunkedTrailer(boost::uint64_t realContentLength, OptionalMap headers) = 0;
		// 报文接收完毕。
		// 如果 onResponseHeaders() 的 contentLength 参数为 CONTENT_TILL_EOF，此处 realContentLength 即为实际接收大小。
		virtual void onLowLevelContentEof(boost::uint64_t realContentLength) = 0;

	public:
		bool send(RequestHeaders requestHeaders, StreamBuffer entity = StreamBuffer());

		// 需要预先对 URI 进行编码处理。
		bool send(Verb verb, std::string uri, OptionalMap getParams = OptionalMap(),
			OptionalMap headers = OptionalMap(), StreamBuffer entity = StreamBuffer());
		bool send(Verb verb, std::string uri, OptionalMap getParams, StreamBuffer entity){
			return send(verb, STD_MOVE(uri), STD_MOVE(getParams), OptionalMap(), STD_MOVE(entity));
		}
	};
}

}

#endif
