// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2018, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_SYSTEM_SESSION_HPP_
#define POSEIDON_SYSTEM_SESSION_HPP_

#include "http/fwd.hpp"
#include "http/session.hpp"

namespace Poseidon {

class SystemServletBase;

class SystemSession : public Http::Session {
private:
	const boost::shared_ptr<const Http::AuthenticationContext> m_auth_ctx;

	bool m_initialized;
	std::string m_decoded_uri;
	boost::shared_ptr<const SystemServletBase> m_servlet;

public:
	SystemSession(Move<UniqueFile> socket, boost::shared_ptr<const Http::AuthenticationContext> auth_ctx);
	~SystemSession();

private:
	void initialize_once(const Http::RequestHeaders &request_headers);

protected:
	void on_sync_expect(Http::RequestHeaders request_headers) OVERRIDE;
	void on_sync_request(Http::RequestHeaders request_headers, StreamBuffer request_entity) OVERRIDE;
};

}

#endif
