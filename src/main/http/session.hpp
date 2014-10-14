#ifndef POSEIDON_HTTP_SESSION_HPP_
#define POSEIDON_HTTP_SESSION_HPP_

#include <string>
#include <cstddef>
#include <boost/shared_ptr.hpp>
#include "../tcp_session_base.hpp"
#include "../optional_map.hpp"
#include "verb.hpp"
#include "status.hpp"

namespace Poseidon {

class HttpSession : public TcpSessionBase {
	friend class HttpServer;
	friend class HttpUpgradedSessionBase;

private:
	enum State {
		ST_FIRST_HEADER,
		ST_HEADERS,
		ST_CONTENTS,
	};

private:
	boost::shared_ptr<const class TimerItem> m_shutdownTimer;

	State m_state;
	std::size_t m_totalLength;
	std::size_t m_contentLength;
	std::string m_line;

	boost::shared_ptr<class HttpUpgradedSessionBase> m_upgradedSession;

	HttpVerb m_verb;
	unsigned m_version;	// x * 10000 + y 表示 HTTP x.y
	std::string m_uri;
	OptionalMap m_getParams;
	OptionalMap m_headers;

public:
	explicit HttpSession(Move<ScopedFile> socket);
	~HttpSession();

private:
	void setRequestTimeout(unsigned long long timeout);

	void onAllHeadersRead();
	void onExpect(const std::string &val);
	void onContentLength(const std::string &val);
	void onUpgrade(const std::string &val);

public:
	void onReadAvail(const void *data, std::size_t size);
	bool shutdown(HttpStatus status);
	bool shutdown(HttpStatus status, OptionalMap headers, StreamBuffer contents);
};

}

#endif
