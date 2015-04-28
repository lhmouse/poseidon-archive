// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2015, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_HTTP_CLIENT_HPP_
#define POSEIDON_HTTP_CLIENT_HPP_

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
	class Client : public TcpClientBase {
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

	public:
		explicit Client(const IpPort &addr, bool useSsl);
		~Client();

	private:
		void onReadAvail(const void *data, std::size_t size) FINAL;

	protected:
		void onReadHup() NOEXCEPT OVERRIDE;

		// 和 Http::Session 不同，这个函数在主线程中调用。
		// 如果 Transfer-Encoding 是 chunked， contentLength 的值为 CONTENT_CHUNKED。
		// 如果没有指定 Content-Length 同时也不是 chunked，contentLength 的值为 CONTENT_TILL_EOF。
		virtual void onResponseHeaders(const ResponseHeaders &responseHeaders, boost::uint64_t contentLength) = 0;
		// 报文可能分几次收到。
		virtual void onEntity(boost::uint64_t contentOffset, const StreamBuffer &entity) = 0;
		// 如果 onResponseHeaders() 的 contentLength 参数为 CONTENT_CHUNKED，使用这个函数标识结束。
		// chunked 允许追加报头。
		virtual void onChunkedTrailer(boost::uint64_t realContentLength, const OptionalMap &headers) = 0;
		// 如果 onResponseHeaders() 的 contentLength 参数为 CONTENT_TILL_EOF，使用这个函数标识结束。
		virtual void onContentEof(boost::uint64_t realContentLength) = 0;

	public:
		bool send(RequestHeaders requestHeaders, StreamBuffer entity = StreamBuffer(), bool fin = false);

		// 需要预先对 URI 进行编码处理。
		bool send(Verb verb, std::string uri, OptionalMap getParams = OptionalMap(), OptionalMap headers = OptionalMap(),
			StreamBuffer entity = StreamBuffer(), bool fin = false);
		bool send(Verb verb, std::string uri, OptionalMap getParams, StreamBuffer entity, bool fin = false){
			return send(verb, STD_MOVE(uri), STD_MOVE(getParams), OptionalMap(), STD_MOVE(entity), fin);
		}
	};
}

}

#endif
