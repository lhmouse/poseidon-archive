// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2018, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_HTTP_CLIENT_HPP_
#define POSEIDON_HTTP_CLIENT_HPP_

#include "low_level_client.hpp"

namespace Poseidon {
namespace Http {

class Client : public Low_level_client {
private:
	class Sync_job_base;
	class Connect_job;
	class Read_hup_job;
	class Response_job;

private:
	Response_headers m_response_headers;
	Stream_buffer m_entity;

public:
	explicit Client(const Sock_addr &addr, bool use_ssl = false, bool verify_peer = true);
	~Client();

protected:
	const Response_headers &get_low_level_response_headers() const {
		return m_response_headers;
	}
	const Stream_buffer &get_low_level_entity() const {
		return m_entity;
	}

	// Tcp_client_base
	void on_connect() OVERRIDE;
	void on_read_hup() OVERRIDE;

	// Low_level_client
	void on_low_level_response_headers(Response_headers response_headers, boost::uint64_t content_length) OVERRIDE;
	void on_low_level_response_entity(boost::uint64_t entity_offset, Stream_buffer entity) OVERRIDE;
	boost::shared_ptr<Upgraded_session_base> on_low_level_response_end(boost::uint64_t content_length, Option_map headers) OVERRIDE;

	// 可覆写。
	virtual void on_sync_connect();

	virtual void on_sync_response(Response_headers response_headers, Stream_buffer entity) = 0;
};

}
}

#endif
