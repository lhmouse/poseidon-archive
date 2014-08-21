#include "../precompiled.hpp"
#include "socket_server_base.hpp"
#include <boost/bind.hpp>
#include "log.hpp"
#include "atomic.hpp"
#include "exception.hpp"
#include "singletons/session_manager.hpp"
#include "tcp_peer.hpp"
#include <arpa/inet.h>
#include <sys/ioctl.h>
#include <errno.h>
using namespace Poseidon;

bool SocketServerBase::tryAccept(boost::shared_ptr<const SocketServerBase> server){
	if(!atomicLoad(server->m_running)){
		return false;
	}
	unsigned char sa[INET_ADDRSTRLEN | INET6_ADDRSTRLEN];
	::socklen_t salen = sizeof(sa);
	ScopedFile client(::accept(server->m_listen.get(), (::sockaddr *)sa, &salen));
	if(!client){
		return true;
	}
	AUTO(peer, server->onClientConnected(client));
	if(!peer){
		return true;
	}
	LOG_INFO <<"Client " <<peer->getRemoteHost() <<" has connected.";
	const int FALSE_VALUE = false;
	if(::ioctl(peer->getFd(), FIONBIO, &FALSE_VALUE) < 0){
		DEBUG_THROW(SystemError, errno);
	}
	SessionManager::registerTcpPeer(peer);
	return true;
}

SocketServerBase::SocketServerBase(const std::string &bindAddr)
	: m_bindAddr(bindAddr), m_running(false)
{
	LOG_INFO <<"Creating socket server on " <<m_bindAddr;

	unsigned char sa[INET_ADDRSTRLEN | INET6_ADDRSTRLEN];
	::socklen_t salen;
	if(::inet_pton(AF_INET, m_bindAddr.c_str(), sa) == 1){
		salen = INET_ADDRSTRLEN;
	} else if(::inet_pton(AF_INET6, m_bindAddr.c_str(), sa) == 1){
		salen = INET6_ADDRSTRLEN;
	} else {
		DEBUG_THROW(SystemError, EINVAL);
	}
	m_listen.reset(::socket(((::sockaddr *)sa)->sa_family, SOCK_STREAM, IPPROTO_TCP));
	if(!m_listen){
		DEBUG_THROW(SystemError, errno);
	}
	const int TRUE_VALUE = true;
	if(::ioctl(m_listen.get(), FIONBIO, &TRUE_VALUE) < 0){
		DEBUG_THROW(SystemError, errno);
	}
	if(::bind(m_listen.get(), (::sockaddr *)sa, salen)){
		DEBUG_THROW(SystemError, errno);
	}
	if(::listen(m_listen.get(), SOMAXCONN)){
		DEBUG_THROW(SystemError, errno);
	}
}
SocketServerBase::~SocketServerBase(){
	LOG_INFO <<"Destroyed socket server on " <<m_bindAddr;
}

void SocketServerBase::start(){
	if(atomicExchange(m_running, true) != false){
		return;
	}
	SessionManager::registerIdleCallback(boost::bind(&tryAccept, shared_from_this()));
}
void SocketServerBase::stop(){
	atomicStore(m_running, false);
}
