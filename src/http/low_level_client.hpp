// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2016, LH_Mouse. All wrongs reserved.

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
	class LowLevelClient : public TcpClientBase, private ClientReader, private ClientWriter {
	protected:
		LowLevelClient(const SockAddr &addr, bool use_ssl);
		LowLevelClient(const IpPort &addr, bool use_ssl);
		~LowLevelClient();

	protected:
		// TcpClientBase
		void on_read_hup() NOEXCEPT OVERRIDE;

		void on_read_avail(StreamBuffer data) OVERRIDE;

		// ClientReader
		void on_response_headers(ResponseHeaders response_headers, std::string transfer_encoding, boost::uint64_t content_length) OVERRIDE;
		void on_response_entity(boost::uint64_t entity_offset, bool is_chunked, StreamBuffer entity) OVERRIDE;
		bool on_response_end(boost::uint64_t content_length, bool is_chunked, OptionalMap headers) OVERRIDE;

		// ClientWriter
		long on_encoded_data_avail(StreamBuffer encoded) OVERRIDE;

		// 可覆写。
		virtual void on_low_level_response_headers(ResponseHeaders response_headers,
			std::string transfer_encoding, boost::uint64_t content_length) = 0;
		virtual void on_low_level_response_entity(boost::uint64_t entity_offset, bool is_chunked, StreamBuffer entity) = 0;
		virtual bool on_low_level_response_end(boost::uint64_t content_length, bool is_chunked, OptionalMap headers) = 0;

	public:
		bool send_headers(RequestHeaders request_headers);
		bool send_entity(StreamBuffer data);

		bool send(RequestHeaders request_headers, StreamBuffer entity = StreamBuffer());

		bool send_chunked_header(RequestHeaders request_headers);
		bool send_chunk(StreamBuffer entity);
		bool send_chunked_trailer(OptionalMap headers);
	};
}

}

#endif
