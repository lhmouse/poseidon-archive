#include "../precompiled.hpp"
#include "player_session.hpp"
#include "log.hpp"
using namespace Poseidon;

PlayerSession::PlayerSession(ScopedFile &socket)
	: TcpPeer(socket)
{
}

void PlayerSession::onReadAvail(const void *data, std::size_t size){
	LOG_DEBUG <<"Received " <<std::string((const char *)data, size);
	send("meow!!!", 7);
}
