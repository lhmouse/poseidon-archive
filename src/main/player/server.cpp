#include "../../precompiled.hpp"
#include "server.hpp"
#include "session.hpp"
using namespace Poseidon;

PlayerServer::PlayerServer(const std::string &bindAddr, unsigned bindPort,
	const std::string &cert, const std::string &privateKey)
	: TcpServerBase(bindAddr, bindPort, cert, privateKey)
{
}

boost::shared_ptr<TcpSessionBase> PlayerServer::onClientConnect(Move<ScopedFile> client) const {
	return boost::make_shared<PlayerSession>(STD_MOVE(client));
}
