// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2017, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_HTTP_CLIENT_HPP_
#define POSEIDON_HTTP_CLIENT_HPP_

#include "low_level_client.hpp"

namespace Poseidon {

namespace Http {
	class Client : public LowLevelClient {
	private:
		class SyncJobBase;
		class ConnectJob;
		class ReadHupJob;
		class ResponseJob;

	private:
		ResponseHeaders m_response_headers;
		StreamBuffer m_entity;

	public:
		explicit Client(const SockAddr &addr, bool use_ssl = false, bool verify_peer = true);
		~Client();

	protected:
		const ResponseHeaders &get_low_level_response_headers() const {
			return m_response_headers;
		}
		const StreamBuffer &get_low_level_entity() const {
			return m_entity;
		}

		// TcpClientBase
		void on_connect() OVERRIDE;
		void on_read_hup() OVERRIDE;

		// LowLevelClient
		void on_low_level_response_headers(ResponseHeaders response_headers, boost::uint64_t content_length) OVERRIDE;
		void on_low_level_response_entity(boost::uint64_t entity_offset, StreamBuffer entity) OVERRIDE;
		boost::shared_ptr<UpgradedSessionBase> on_low_level_response_end(boost::uint64_t content_length, OptionalMap headers) OVERRIDE;

		// 可覆写。
		virtual void on_sync_connect();

		virtual void on_sync_response(ResponseHeaders response_headers, StreamBuffer entity) = 0;
	};
}

}

#endif
