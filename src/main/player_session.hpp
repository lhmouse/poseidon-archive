#ifndef POSEIDON_PLAYER_SESSION_HPP_
#define POSEIDON_PLAYER_SESSION_HPP_

#include "tcp_peer.hpp"

namespace Poseidon {

class PlayerSession : public TcpPeer {
private:

public:
	explicit PlayerSession(ScopedFile &socket);

protected:
	void onReadAvail(const void *data, std::size_t size);
};

}

#endif
