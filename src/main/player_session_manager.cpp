#include "../precompiled.hpp"
#include "player_session_manager.hpp"
#include "player_session.hpp"
#include "log.hpp"
#include <boost/make_shared.hpp>
#include <boost/ref.hpp>
using namespace Poseidon;

PlayerSessionManager::PlayerSessionManager(const std::string &bindAddr, unsigned bindPort)
	: SocketServerBase(bindAddr, bindPort)
{
}

boost::shared_ptr<TcpPeer> PlayerSessionManager::onClientConnect(ScopedFile &client) const {
	AUTO(ps, boost::make_shared<PlayerSession>(boost::ref(client)));

	LOG_INFO <<"Created player session from client " <<ps->getRemoteIp();
//	ps->shutdown();
	return ps;
}
