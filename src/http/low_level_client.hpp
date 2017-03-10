// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2017, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_HTTP_LOW_LEVEL_CLIENT_HPP_
#define POSEIDON_HTTP_LOW_LEVEL_CLIENT_HPP_

#include "../tcp_client_base.hpp"
#include "client_reader.hpp"
#include "client_writer.hpp"
#include "request_headers.hpp"
#include "response_headers.hpp"
#include "status_codes.hpp"

namespace Poseidon {

namespace Http {
	class UpgradedClientBase;
	class HeaderOption;

	class LowLevelClient : public TcpClientBase, private ClientReader, private ClientWriter {
		friend UpgradedClientBase;

	private:
		mutable Mutex m_upgraded_client_mutex;
		boost::shared_ptr<UpgradedClientBase> m_upgraded_client;

	protected:
		LowLevelClient(const SockAddr &addr, bool use_ssl, bool verify_peer);
		LowLevelClient(const IpPort &addr, bool use_ssl, bool verify_peer);
		~LowLevelClient();

	protected:
		const boost::shared_ptr<UpgradedClientBase> &get_low_level_upgraded_client() const {
			// Epoll 线程读取不需要锁。
			return m_upgraded_client;
		}

		// TcpClientBase
		void on_read_hup() NOEXCEPT OVERRIDE;

		void on_receive(StreamBuffer data) OVERRIDE;

		// ClientReader
		void on_response_headers(ResponseHeaders response_headers, boost::uint64_t content_length) OVERRIDE;
		void on_response_entity(boost::uint64_t entity_offset, StreamBuffer entity) OVERRIDE;
		bool on_response_end(boost::uint64_t content_length, OptionalMap headers) OVERRIDE;

		// ClientWriter
		long on_encoded_data_avail(StreamBuffer encoded) OVERRIDE;

		// 可覆写。
		virtual void on_low_level_response_headers(ResponseHeaders response_headers, boost::uint64_t content_length) = 0;
		virtual void on_low_level_response_entity(boost::uint64_t entity_offset, StreamBuffer entity) = 0;
		virtual boost::shared_ptr<UpgradedClientBase> on_low_level_response_end(boost::uint64_t content_length, OptionalMap headers) = 0;

	public:
		boost::shared_ptr<UpgradedClientBase> get_upgraded_client() const;

		bool send(RequestHeaders request_headers, StreamBuffer entity = StreamBuffer());
		bool send(Verb verb, std::string uri, OptionalMap get_params = OptionalMap());
		bool send(Verb verb, std::string uri, OptionalMap get_params, StreamBuffer entity, const HeaderOption &content_type);
		bool send(Verb verb, std::string uri, OptionalMap get_params, OptionalMap headers, StreamBuffer entity = StreamBuffer());

		bool send_chunked_header(RequestHeaders request_headers);
		bool send_chunk(StreamBuffer entity);
		bool send_chunked_trailer(OptionalMap headers);
	};
}

}

#endif
