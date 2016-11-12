// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2016, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_HTTP_SESSION_HPP_
#define POSEIDON_HTTP_SESSION_HPP_

#include "low_level_session.hpp"

namespace Poseidon {

namespace Http {
	class Session : public LowLevelSession {
	private:
		class SyncJobBase;
		class ReadHupJob;
		class ExpectJob;
		class RequestJob;
		class ErrorJob;

	private:
		const boost::uint64_t m_max_request_length;

		boost::uint64_t m_size_total;
		RequestHeaders m_request_headers;
		std::string m_transfer_encoding;
		StreamBuffer m_entity;

	public:
		explicit Session(UniqueFile socket, boost::uint64_t max_request_length = 0);
		~Session();

	protected:
		boost::uint64_t get_low_level_size_total() const {
			return m_size_total;
		}
		const RequestHeaders &get_low_level_request_headers() const {
			return m_request_headers;
		}
		const std::string &get_low_level_transfer_encoding() const {
			return m_transfer_encoding;
		}
		const StreamBuffer &get_low_level_entity() const {
			return m_entity;
		}

		// TcpSessionBase
		void on_read_hup() NOEXCEPT OVERRIDE;

		// LowLevelSession
		void on_read_avail(StreamBuffer data) OVERRIDE;

		void on_low_level_request_headers(RequestHeaders request_headers,
			std::string transfer_encoding, boost::uint64_t content_length) OVERRIDE;
		void on_low_level_request_entity(boost::uint64_t entity_offset, bool is_chunked, StreamBuffer entity) OVERRIDE;
		boost::shared_ptr<UpgradedSessionBase> on_low_level_request_end(
			boost::uint64_t content_length, bool is_chunked, OptionalMap headers) OVERRIDE;

		// 可覆写。
		virtual void on_sync_request(RequestHeaders request_headers, StreamBuffer entity) = 0;
	};
}

}

#endif
