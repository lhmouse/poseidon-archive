#include "../precompiled.hpp"
#include "http_session.hpp"
#include "log.hpp"
using namespace Poseidon;

HttpSession::HttpSession(ScopedFile &socket)
	: TcpPeer(socket)
{
}

void HttpSession::onReadAvail(const void *data, std::size_t size){
	LOG_DEBUG("Received ", std::string((const char *)data, size));
}
void HttpSession::perform() const {
}
