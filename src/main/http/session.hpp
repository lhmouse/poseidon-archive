#ifndef POSEIDON_HTTP_SESSION_HPP_
#define POSEIDON_HTTP_SESSION_HPP_

#include <string>
#include <cstddef>
#include <boost/shared_ptr.hpp>
#include <boost/weak_ptr.hpp>
#include "../tcp_session_base.hpp"
#include "../optional_map.hpp"
#include "verb.hpp"

namespace Poseidon {

class HttpSession : public TcpSessionBase {
	friend class HttpServer;

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

	HttpVerb m_verb;
	std::string m_uri;
	OptionalMap m_getParams;
	OptionalMap m_headers;

public:
	explicit HttpSession(Move<ScopedFile> socket);
	~HttpSession();

private:
	void resetTimeout(unsigned long long timeout);

	void onAllHeadersRead();

protected:
	void onReadAvail(const void *data, std::size_t size);
};

}

#endif
