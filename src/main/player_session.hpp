#ifndef POSEIDON_PLAYER_SESSION_HPP_
#define POSEIDON_PLAYER_SESSION_HPP_

#include "tcp_peer.hpp"
#include "singletons/job_dispatcher.hpp"
#include <vector>
#include <cstddef>

namespace Poseidon {

class PlayerSession : public TcpPeer, public JobBase {
private:
	std::vector<unsigned char> m_received;

	long m_payloadLen;
	unsigned m_protocolId;

public:
	explicit PlayerSession(ScopedFile &socket);

private:
	void onReadAvail(const void *data, std::size_t size);
	void perform() const;

public:
};

}

#endif
