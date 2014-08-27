#ifndef POSEIDON_HTTP_SESSION_HPP_
#define POSEIDON_HTTP_SESSION_HPP_

#include <cstddef>
#include "tcp_peer.hpp"
#include "stream_buffer.hpp"
#include "singletons/job_dispatcher.hpp"

namespace Poseidon {

class HttpSession : public TcpPeer, public JobBase {
private:
	StreamBuffer m_received;


public:
	explicit HttpSession(ScopedFile &socket);

private:
	void onReadAvail(const void *data, std::size_t size);
	void perform() const;
};

}

#endif
