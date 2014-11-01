#include "../precompiled.hpp"
#include "server.hpp"
#include "session.hpp"
using namespace Poseidon;

PlayerServer::PlayerServer(std::size_t category, std::string bindAddr, unsigned bindPort,
	const char *cert, const char *privateKey)
	: TcpServerBase(STD_MOVE(bindAddr), bindPort, cert, privateKey)
	, m_category(category)
{
}

boost::shared_ptr<TcpSessionBase> PlayerServer::onClientConnect(ScopedFile client) const {
	return boost::make_shared<PlayerSession>(m_category, STD_MOVE(client));
}
