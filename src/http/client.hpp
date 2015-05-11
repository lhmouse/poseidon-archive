// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2015, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_HTTP_CLIENT_HPP_
#define POSEIDON_HTTP_CLIENT_HPP_

#include "low_level_client.hpp"

namespace Poseidon {

namespace Http {
	class Client : public LowLevelClient {
	private:
		class ResponseHeaderJob;
		class EntityJob;
		class ContentEofJob;
		class ChunkedTrailerJob;

	protected:
		Client(const SockAddr &addr, bool useSsl);
		Client(const IpPort &addr, bool useSsl);
		~Client();

	protected:
		void onLowLevelResponseHeaders(ResponseHeaders responseHeaders, boost::uint64_t contentLength) OVERRIDE;
		void onLowLevelEntity(boost::uint64_t contentOffset, StreamBuffer entity) OVERRIDE;
		void onLowLevelChunkedTrailer(boost::uint64_t realContentLength, OptionalMap headers) OVERRIDE;
		void onLowLevelContentEof(boost::uint64_t realContentLength) OVERRIDE;

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
	};
}

}

#endif
