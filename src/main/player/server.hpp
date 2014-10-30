#ifndef POSEIDON_PLAYER_SERVER_HPP_
#define POSEIDON_PLAYER_SERVER_HPP_

#include "../tcp_server_base.hpp"

namespace Poseidon {

class PlayerServer : public TcpServerBase {
private:
	const std::size_t m_category;

public:
	PlayerServer(std::size_t category, std::string bindAddr, unsigned bindPort,
		const SharedNtmbs &cert, const SharedNtmbs &privateKey);

protected:
	boost::shared_ptr<class TcpSessionBase> onClientConnect(ScopedFile client) const;
};

}

#endif
