#ifndef POSEIDON_HTTP_SESSION_HPP_
#define POSEIDON_HTTP_SESSION_HPP_

#include <cstddef>
#include <string>
#include "tcp_session_base.hpp"
#include "stream_buffer.hpp"
#include "optional_map.hpp"
#include "singletons/job_dispatcher.hpp"
#include "http_status.hpp"

namespace Poseidon {

class HttpSession : public TcpSessionBase, public JobBase {
private:
	enum State {
		ST_FIRST_HEADER,
		ST_MORE_HEADERS,
		ST_CONTENTS
	};

private:
	StreamBuffer m_received;
	State m_state;

	std::string m_verb;
	std::string m_uri;
	std::string m_userAgent;
	OptionalMap m_getParams;
	OptionalMap m_postParams;

public:
	explicit HttpSession(ScopedFile &socket);

protected:
	// TcpSessionBase
	void onReadAvail(const void *data, std::size_t size);
	void onRemoteClose();

	// JobBase
	void perform() const;
};

}

#endif
