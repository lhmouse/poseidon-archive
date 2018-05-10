// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2018, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_SYSTEM_HTTP_SESSION_HPP_
#define POSEIDON_SYSTEM_HTTP_SESSION_HPP_

#include "http/fwd.hpp"
#include "http/session.hpp"

namespace Poseidon {

class System_http_servlet_base;

class System_http_session : public Http::Session {
private:
	const boost::shared_ptr<const Http::Authentication_context> m_auth_ctx;

	bool m_initialized;
	std::string m_decoded_uri;
	boost::shared_ptr<const System_http_servlet_base> m_servlet;

public:
	System_http_session(Move<Unique_file> socket, boost::shared_ptr<const Http::Authentication_context> auth_ctx);
	~System_http_session();

private:
	void initialize_once(const Http::Request_headers &request_headers);

protected:
	void on_sync_expect(Http::Request_headers request_headers) OVERRIDE;
	void on_sync_request(Http::Request_headers request_headers, Stream_buffer request_entity) OVERRIDE;
};

}

#endif
