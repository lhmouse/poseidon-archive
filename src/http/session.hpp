// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2015, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_HTTP_SESSION_HPP_
#define POSEIDON_HTTP_SESSION_HPP_

#include "../cxx_ver.hpp"
#include <string>
#include <cstddef>
#include <boost/shared_ptr.hpp>
#include <boost/cstdint.hpp>
#include "request_headers.hpp"
#include "response_headers.hpp"
#include "status_codes.hpp"
#include "../tcp_session_base.hpp"
#include "../mutex.hpp"
#include "../stream_buffer.hpp"

namespace Poseidon {

namespace Http {
	class UpgradedSessionBase;

	class Session : public TcpSessionBase {
		friend UpgradedSessionBase;

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
		};

	private:
		class ContinueJob;
		class RequestJob;
		class ErrorJob;

	private:
		mutable Mutex m_upgradedSessionMutex;
		boost::shared_ptr<UpgradedSessionBase> m_upgradedSession;

		StreamBuffer m_received;

		boost::uint64_t m_sizeTotal;
		bool m_expectingNewLine;
		boost::uint64_t m_sizeExpecting;
		State m_state;

		RequestHeaders m_requestHeaders;
		StreamBuffer m_chunkedEntity;

	public:
		explicit Session(UniqueFile socket);
		~Session();

	protected:
		void onReadAvail(const void *data, std::size_t size) OVERRIDE;

		void onReadHup() NOEXCEPT OVERRIDE;
		void onWriteHup() NOEXCEPT OVERRIDE;
		void onClose() NOEXCEPT OVERRIDE;

		// 和 Http::Client 不同，这个函数在 Epoll 线程中调用。
		// 如果 Transfer-Encoding 是 chunked， contentLength 的值为 CONTENT_CHUNKED。
		virtual boost::shared_ptr<UpgradedSessionBase> onRequestHeaders(
			RequestHeaders &requestHeaders, boost::uint64_t contentLength);
		virtual void onRequest(
			const RequestHeaders &requestHeaders, const StreamBuffer &entity) = 0;

	public:
		boost::shared_ptr<UpgradedSessionBase> getUpgradedSession() const;

		bool send(ResponseHeaders responseHeaders, StreamBuffer entity = StreamBuffer());

		bool send(StatusCode statusCode, OptionalMap headers = OptionalMap(), StreamBuffer entity = StreamBuffer());
		bool send(StatusCode statusCode, StreamBuffer entity){
			return send(statusCode, OptionalMap(), STD_MOVE(entity));
		}

		bool sendDefault(StatusCode statusCode, OptionalMap headers);
		bool sendDefault(StatusCode statusCode){
			return sendDefault(statusCode, OptionalMap());
		}
	};
}

}

#endif
