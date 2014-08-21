#ifndef POSEIDON_PLAYER_SESSION_MANAGER_HPP_
#define POSEIDON_PLAYER_SESSION_MANAGER_HPP_

#include "socket_server_base.hpp"
#include "player_session.hpp"

namespace Poseidon {

class PlayerSessionManager : public SocketServerBase {
public:
	PlayerSessionManager(const std::string &bindAddr, unsigned bindPort);

protected:
	boost::shared_ptr<TcpPeer> onClientConnect(ScopedFile &client) const;
};

}

#endif
