#ifndef POSEIDON_HTTP_SESSION_HPP_
#define POSEIDON_HTTP_SESSION_HPP_

#include "../../cxx_ver.hpp"
#include <string>
#include <set>
#include <cstddef>
#include <boost/shared_ptr.hpp>
#include "../tcp_session_base.hpp"
#include "../optional_map.hpp"
#include "verb.hpp"
#include "status.hpp"

namespace Poseidon {

class HttpUpgradedSessionBase;

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

	boost::shared_ptr<HttpUpgradedSessionBase> m_upgradedSession;
	boost::shared_ptr<std::set<std::string> > m_authInfo;

	HttpVerb m_verb;
	unsigned m_version;	// x * 10000 + y 表示 HTTP x.y
	std::string m_uri;
	OptionalMap m_getParams;
	OptionalMap m_headers;

public:
	explicit HttpSession(Move<ScopedFile> socket);
	~HttpSession();

private:
	void onReadAvail(const void *data, std::size_t size);

	void setRequestTimeout(unsigned long long timeout);

	void onAllHeadersRead();
	void onExpect(const std::string &val);
	void onContentLength(const std::string &val);
	void onUpgrade(const std::string &val);
	void onAuthorization(const std::string &val);

public:
	void setAuthInfo(boost::shared_ptr<std::set<std::string> > authInfo){
		m_authInfo.swap(authInfo);
	}

	bool send(HttpStatus status, OptionalMap headers, StreamBuffer contents, bool final = false);
	bool send(HttpStatus status, StreamBuffer contents = StreamBuffer(), bool final = false){
		return send(status, OptionalMap(), STD_MOVE(contents), final);
	}

	bool sendDefault(HttpStatus status, OptionalMap headers, bool final = false);
	bool sendDefault(HttpStatus status, bool final = false){
		return sendDefault(status, OptionalMap(), final);
	}
};

}

#endif
