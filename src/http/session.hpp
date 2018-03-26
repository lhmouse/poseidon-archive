// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2018, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_HTTP_SESSION_HPP_
#define POSEIDON_HTTP_SESSION_HPP_

#include "low_level_session.hpp"

namespace Poseidon {
namespace Http {

class Session : public Low_level_session {
private:
	class Sync_job_base;
	class Read_hup_job;
	class Expect_job;
	class Request_job;
	class Error_job;

private:
	volatile boost::uint64_t m_max_request_length;
	boost::uint64_t m_size_total;
	Request_headers m_request_headers;
	Stream_buffer m_entity;

public:
	explicit Session(Move<Unique_file> socket);
	~Session();

protected:
	boost::uint64_t get_low_level_size_total() const {
		return m_size_total;
	}
	const Request_headers &get_low_level_request_headers() const {
		return m_request_headers;
	}
	const Stream_buffer &get_low_level_entity() const {
		return m_entity;
	}

	// Tcp_session_base
	void on_read_hup() OVERRIDE;

	// Low_level_session
	void on_low_level_request_headers(Request_headers request_headers, boost::uint64_t content_length) OVERRIDE;
	void on_low_level_request_entity(boost::uint64_t entity_offset, Stream_buffer entity) OVERRIDE;
	boost::shared_ptr<Upgraded_session_base> on_low_level_request_end(boost::uint64_t content_length, Optional_map headers) OVERRIDE;

	// 可覆写。
	virtual void on_sync_expect(Request_headers request_headers);
	virtual void on_sync_request(Request_headers request_headers, Stream_buffer entity) = 0;

public:
	boost::uint64_t get_max_request_length() const;
	void set_max_request_length(boost::uint64_t max_request_length);
};

}
}

#endif
