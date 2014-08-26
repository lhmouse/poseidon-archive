#include "../precompiled.hpp"
#include "player_session.hpp"
#include "log.hpp"
#include "vint50.hpp"
using namespace Poseidon;


#include "profiler.hpp"

__attribute__((__noinline__)) void baz(){
	PROFILE_ME;
	::usleep(10000);
}
__attribute__((__noinline__)) void bar(){
	PROFILE_ME;
	::usleep(20000);
	baz();
	::usleep(10000);
}
__attribute__((__noinline__)) void foo(){
	PROFILE_ME;
	::usleep(10000);
	bar();
	::usleep(20000);
}

PlayerSession::PlayerSession(ScopedFile &socket)
	: TcpPeer(socket), m_payloadLen(-1)
{
}

void PlayerSession::onReadAvail(const void *data, std::size_t size){
	LOG_DEBUG("Received ", std::string((const char *)data, size));
/*	send("meow meow meow!!!", 17);
	unsigned char tmp[7];
	unsigned long long ll = 0x12345, ll2;
	unsigned char *write = tmp, *read = tmp;
	vuint50ToBinary(ll, write);
	vuint50FromBinary(ll2, read, write);
	LOG_DEBUG("read = ", (void *)read, ", write = ", (void *)write, ", serialized = ", (write - tmp), ", ll2 = ", std::hex, ll2);*/
	for(int i = 0; i < 10; ++i){
		char data = '0' + i;
		send(&data, 1);
	}
	foo();
	AUTO(ss, Profiler::snapshot());
	for(AUTO(it, ss.begin()); it != ss.end(); ++it){
		LOG_WARNING("file = ", it->file, ", line = ", it->line, ", samples = ", it->samples,
			", usTotal = ", it->usTotal, ", usExclusive = ", it->usExclusive);
	}
}
void PlayerSession::perform() const {
}
