// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2015, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_HTTP_LOW_LEVEL_SESSION_HPP_
#define POSEIDON_HTTP_LOW_LEVEL_SESSION_HPP_

#include "../cxx_ver.hpp"
#include <string>
#include <vector>
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
	class UpgradedLowLevelSessionBase;

	class LowLevelSession : public TcpSessionBase {
		friend UpgradedLowLevelSessionBase;

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
		mutable Mutex m_upgradedSessionMutex;
		boost::shared_ptr<UpgradedLowLevelSessionBase> m_upgradedSession;

		StreamBuffer m_received;

		boost::uint64_t m_sizeTotal;
		bool m_expectingNewLine;
		boost::uint64_t m_sizeExpecting;
		State m_state;

		RequestHeaders m_requestHeaders;
		StreamBuffer m_chunkedEntity;

	public:
		explicit LowLevelSession(UniqueFile socket);
		~LowLevelSession();

	protected:
		void onReadAvail(const void *data, std::size_t size) OVERRIDE;

		void onReadHup() NOEXCEPT OVERRIDE;
		void onWriteHup() NOEXCEPT OVERRIDE;
		void onClose(int errCode) NOEXCEPT OVERRIDE;

		// transferEncoding 确保已被转换为小写、已排序，并且 chunked 和 identity 被移除（如果有的话）。
		virtual boost::shared_ptr<UpgradedLowLevelSessionBase> onLowLevelRequestHeaders(RequestHeaders &requestHeaders,
			const std::vector<std::string> &transferEncoding, boost::uint64_t contentLength) = 0;

		virtual void onLowLevelRequest(RequestHeaders requestHeaders, StreamBuffer entity) = 0;
		virtual void onLowLevelError(StatusCode statusCode, OptionalMap headers) = 0;

	public:
		boost::shared_ptr<UpgradedLowLevelSessionBase> getUpgradedSession() const;

		bool send(ResponseHeaders responseHeaders, StreamBuffer entity = StreamBuffer());

		bool send(StatusCode statusCode, OptionalMap headers = OptionalMap(), StreamBuffer entity = StreamBuffer());
		bool send(StatusCode statusCode, StreamBuffer entity){
			return send(statusCode, OptionalMap(), STD_MOVE(entity));
		}

		bool sendDefault(StatusCode statusCode, OptionalMap headers = OptionalMap());
	};
}

}

#endif
