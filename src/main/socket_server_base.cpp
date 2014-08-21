#include "../precompiled.hpp"
#include "socket_server_base.hpp"
#include <boost/bind.hpp>
#include "log.hpp"
#include "atomic.hpp"
#include "exception.hpp"
#include "singletons/epoll_dispatcher.hpp"
#include "tcp_peer.hpp"
#include <arpa/inet.h>
#include <sys/ioctl.h>
#include <errno.h>
using namespace Poseidon;

bool SocketServerBase::tryAccept(boost::shared_ptr<const SocketServerBase> server){
	if(!atomicLoad(server->m_running)){
		return false;
	}

	ScopedFile client(::accept(server->m_listen.get(), NULL, NULL));
	if(!client){
		return true;
	}
	AUTO(peer, server->onClientConnect(client));
	if(!peer){
		return true;
	}
	EpollDispatcher::registerTcpPeer(peer);
	LOG_INFO <<"Client '" <<peer->getRemoteIp() <<"' has connected.";
	return true;
}

SocketServerBase::SocketServerBase(const std::string &bindAddr, unsigned bindPort)
	: m_running(false)
{
	union {
		::sockaddr sa;
		::sockaddr_in sin;
		::sockaddr_in6 sin6;
	} u;
	::socklen_t salen = sizeof(u);

	m_bindAddr.resize(63);
	const char *text;
	if(::inet_pton(AF_INET, bindAddr.c_str(), &u.sin.sin_addr) == 1){
		u.sin.sin_family = AF_INET;
		u.sin.sin_port = ::htons(bindPort);
		salen = sizeof(::sockaddr_in);
		text = ::inet_ntop(AF_INET, &u.sin.sin_addr, &m_bindAddr[0], m_bindAddr.size());
	} else if(::inet_pton(AF_INET6, bindAddr.c_str(), &u.sin6.sin6_addr) == 1){
		u.sin6.sin6_family = AF_INET6;
		u.sin6.sin6_port = ::htons(bindPort);
		salen = sizeof(::sockaddr_in6);
		text = ::inet_ntop(AF_INET6, &u.sin6.sin6_addr, &m_bindAddr[0], m_bindAddr.size());
	} else {
		DEBUG_THROW(Exception, "Unknown address format. IP expected.");
	}
	if(!text){
		DEBUG_THROW(SystemError, errno);
	}
	m_bindAddr.resize(std::strlen(text));
	m_bindAddr += ':';
	m_bindAddr += boost::lexical_cast<std::string>(bindPort);

	LOG_INFO <<"Creating socket server on " <<m_bindAddr <<"...";

	m_listen.reset(::socket(u.sa.sa_family, SOCK_STREAM, IPPROTO_TCP));
	if(!m_listen){
		const int code = errno;
		LOG_ERROR <<"Error creating socket.";
		DEBUG_THROW(SystemError, code);
	}
	const int TRUE_VALUE = true;
	if(::ioctl(m_listen.get(), FIONBIO, &TRUE_VALUE) < 0){
		const int code = errno;
		LOG_ERROR <<"Could not set listen socket to non-block mode.";
		DEBUG_THROW(SystemError, code);
	}
	if(::bind(m_listen.get(), &u.sa, salen)){
		const int code = errno;
		LOG_ERROR <<"Could not bind socket onto the specified address.";
		DEBUG_THROW(SystemError, code);
	}
	if(::listen(m_listen.get(), SOMAXCONN)){
		const int code = errno;
		LOG_ERROR <<"Could not listen on socket.";
		DEBUG_THROW(SystemError, code);
	}
}
SocketServerBase::~SocketServerBase(){
	LOG_INFO <<"Destroyed socket server on " <<m_bindAddr;
}

void SocketServerBase::start(){
	if(atomicExchange(m_running, true) != false){
		return;
	}
	EpollDispatcher::registerIdleCallback(boost::bind(&tryAccept, shared_from_this()));
}
void SocketServerBase::stop(){
	atomicStore(m_running, false);
}
