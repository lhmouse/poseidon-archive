#include "../precompiled.hpp"
#include "player_session.hpp"
#include "log.hpp"
#include "vint50.hpp"
using namespace Poseidon;

PlayerSession::PlayerSession(ScopedFile &socket)
	: TcpPeer(socket), m_payloadLen(-1)
{
}

void PlayerSession::onReadAvail(const void *data, std::size_t size){
	LOG_DEBUG, "Received ", std::string((const char *)data, size);
/*	send("meow meow meow!!!", 17);
	unsigned char tmp[7];
	unsigned long long ll = 0x12345, ll2;
	unsigned char *write = tmp, *read = tmp;
	vuint50ToBinary(ll, write);
	vuint50FromBinary(ll2, read, write);
	LOG_DEBUG, "read = ", (void *)read, ", write = ", (void *)write, ", serialized = ", (write - tmp), ", ll2 = ", std::hex, ll2;*/
	for(int i = 0; i < 6553600; ++i){
		send("\0", 1);
	}
}
void PlayerSession::perform() const {
}
