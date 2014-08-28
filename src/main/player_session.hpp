#ifndef POSEIDON_PLAYER_SESSION_HPP_
#define POSEIDON_PLAYER_SESSION_HPP_

#include <cstddef>
#include "tcp_session_base.hpp"
#include "stream_buffer.hpp"
#include "singletons/job_dispatcher.hpp"

namespace Poseidon {

class PlayerSession : public TcpSessionBase, public JobBase {
private:
	StreamBuffer m_received;

	long m_payloadLen;
	unsigned m_protocolId;

public:
	explicit PlayerSession(ScopedFile &socket);

private:
	void onReadAvail(const void *data, std::size_t size);
	void onReadComplete();

	void perform() const;
};

}

#endif
