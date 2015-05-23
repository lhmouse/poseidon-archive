// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2015, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_HTTP_SESSION_HPP_
#define POSEIDON_HTTP_SESSION_HPP_

#include "../mutex.hpp"
#include "../tcp_session_base.hpp"
#include "server_reader.hpp"
#include "server_writer.hpp"
#include "request_headers.hpp"
#include "response_headers.hpp"
#include "status_codes.hpp"

namespace Poseidon {

namespace Http {
	class UpgradedSessionBase;

	class Session : public TcpSessionBase, private ServerReader, private ServerWriter {
	private:
		class SyncJobBase;
		class ContinueJob;
		class RequestJob;
		class ErrorJob;

	private:
		boost::uint64_t m_sizeTotal;
		RequestHeaders m_requestHeaders;
		std::string m_transferEncoding;
		StreamBuffer m_entity;

		mutable Mutex m_upgradedSessionMutex;
		boost::shared_ptr<UpgradedSessionBase> m_upgradedSession;

	public:
		explicit Session(UniqueFile socket);
		~Session();

	protected:
		// TcpSessionBase
		void onReadHup() NOEXCEPT OVERRIDE;
		void onClose(int errCode) NOEXCEPT OVERRIDE;

		void onReadAvail(StreamBuffer data) OVERRIDE;

		// ServerReader
		void onRequestHeaders(RequestHeaders requestHeaders, std::string transferEncoding, boost::uint64_t contentLength) OVERRIDE;
		void onRequestEntity(boost::uint64_t entityOffset, bool isChunked, StreamBuffer entity) OVERRIDE;
		bool onRequestEnd(boost::uint64_t contentLength, bool isChunked, OptionalMap headers) OVERRIDE;

		// ServerWriter
		long onEncodedDataAvail(StreamBuffer encoded) OVERRIDE;

		// 可覆写。
		virtual boost::shared_ptr<UpgradedSessionBase> predispatchRequest(RequestHeaders &requestHeaders, StreamBuffer &entity);

		virtual void onSyncRequest(const RequestHeaders &requestHeaders, const StreamBuffer &entity) = 0;

	public:
		boost::shared_ptr<UpgradedSessionBase> getUpgradedSession() const;

		bool send(ResponseHeaders responseHeaders, StreamBuffer entity = StreamBuffer());
		bool send(StatusCode statusCode, OptionalMap headers = OptionalMap(), StreamBuffer entity = StreamBuffer());
		bool sendDefault(StatusCode statusCode, OptionalMap headers = OptionalMap());

		bool send(StatusCode statusCode, StreamBuffer entity){
			return send(statusCode, OptionalMap(), STD_MOVE(entity));
		}
	};
}

}

#endif
