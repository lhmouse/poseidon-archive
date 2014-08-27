#ifndef POSEIDON_HTTP_SESSION_HPP_
#define POSEIDON_HTTP_SESSION_HPP_

#include <cstddef>
#include <string>
#include "tcp_session_base.hpp"
#include "stream_buffer.hpp"
#include "optional_map.hpp"
#include "singletons/job_dispatcher.hpp"

namespace Poseidon {

class HttpSession : public TcpSessionBase, public JobBase {
private:
	StreamBuffer m_received;

	std::string m_uri;
	OptionalMap m_getParams;
	std::string m_contents;

public:
	explicit HttpSession(ScopedFile &socket);

private:
	void onReadAvail(const void *data, std::size_t size);
	

	void perform() const;
};

}

#endif
