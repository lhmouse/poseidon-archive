// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2015, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_HTTP_CLIENT_HPP_
#define POSEIDON_HTTP_CLIENT_HPP_

#include "../tcp_client_base.hpp"
#include "client_reader.hpp"
#include "client_writer.hpp"
#include "request_headers.hpp"
#include "response_headers.hpp"
#include "status_codes.hpp"

namespace Poseidon {

namespace Http {
	class Client : public TcpClientBase, private ClientReader, private ClientWriter {
	private:
		class SyncJobBase;
		class ConnectJob;
		class ResponseHeadersJob;
		class ResponseEntityJob;
		class ResponseEndJob;

	protected:
		Client(const SockAddr &addr, bool useSsl);
		Client(const IpPort &addr, bool useSsl);
		~Client();

	protected:
		// TcpClientBase
		void onConnect() OVERRIDE;
		void onReadHup() NOEXCEPT OVERRIDE;

		void onReadAvail(StreamBuffer data) OVERRIDE;

		// ClientReader
		void onResponseHeaders(ResponseHeaders responseHeaders, std::string transferEncoding, boost::uint64_t contentLength) OVERRIDE;
		void onResponseEntity(boost::uint64_t entityOffset, bool isChunked, StreamBuffer entity) OVERRIDE;
		bool onResponseEnd(boost::uint64_t contentLength, bool isChunked, OptionalMap headers) OVERRIDE;

		// ClientWriter
		long onEncodedDataAvail(StreamBuffer encoded) OVERRIDE;

		// 可覆写。
		virtual void onSyncConnect();

		virtual void onSyncResponseHeaders(ResponseHeaders responseHeaders, std::string transferEncoding, boost::uint64_t contentLength) = 0;
		virtual void onSyncResponseEntity(boost::uint64_t entityOffset, bool isChunked, StreamBuffer entity) = 0;
		virtual void onSyncResponseEnd(boost::uint64_t contentLength, bool isChunked, OptionalMap headers) = 0;
	};
}

}

#endif
