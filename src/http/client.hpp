// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2016, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_HTTP_CLIENT_HPP_
#define POSEIDON_HTTP_CLIENT_HPP_

#include "low_level_client.hpp"

namespace Poseidon {

namespace Http {
	class Client : public LowLevelClient {
	private:
		class SyncJobBase;
		class ConnectJob;
		class ResponseJob;

	private:
		ResponseHeaders m_response_headers;
		std::string m_transfer_encoding;
		StreamBuffer m_entity;

	protected:
		Client(const SockAddr &addr, bool use_ssl);
		Client(const IpPort &addr, bool use_ssl);
		~Client();

	protected:
		// TcpClientBase
		void on_connect() OVERRIDE;

		// LowLevelClient
		void on_low_level_response_headers(
			ResponseHeaders response_headers, std::string transfer_encoding, boost::uint64_t content_length) OVERRIDE;
		void on_low_level_response_entity(boost::uint64_t entity_offset, bool is_chunked, StreamBuffer entity) OVERRIDE;
		bool on_low_level_response_end(boost::uint64_t content_length, bool is_chunked, OptionalMap headers) OVERRIDE;

		// 可覆写。
		virtual void on_sync_connect();

		virtual void on_sync_response(ResponseHeaders response_headers, std::string transfer_encoding, StreamBuffer entity) = 0;
	};
}

}

#endif
