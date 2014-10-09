#ifndef POSEIDON_PLAYER_SERVER_HPP_
#define POSEIDON_PLAYER_SERVER_HPP_

#include "../tcp_server_base.hpp"

namespace Poseidon {

class PlayerServer : public TcpServerBase {
public:
	PlayerServer(const std::string &bindAddr, unsigned bindPort);

protected:
	boost::shared_ptr<class TcpSessionBase>
		onClientConnect(Move<ScopedFile> client) const;
};

}

#endif
