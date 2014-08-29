#ifndef POSEIDON_HTTP_SESSION_HPP_
#define POSEIDON_HTTP_SESSION_HPP_

#include <string>
#include <cstddef>
#include "tcp_session_base.hpp"
#include "optional_map.hpp"
#include "http_verb.hpp"

namespace Poseidon {

class HttpSession : public TcpSessionBase {
private:
	int m_headerIndex;
	std::size_t m_totalLength;
	std::size_t m_contentLength;
	std::string m_line;

	HttpVerb m_verb;
	std::string m_uri;
	OptionalMap m_getParams;
	OptionalMap m_postParams;

public:
	explicit HttpSession(ScopedFile &socket);

protected:
	void onReadAvail(const void *data, std::size_t size);
};

}

#endif
