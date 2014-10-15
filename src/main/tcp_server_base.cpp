#include "../precompiled.hpp"
#include "tcp_server_base.hpp"
#include "tcp_session_base.hpp"
#define POSEIDON_TCP_SESSION_SSL_IMPL_
#include "tcp_session_ssl_impl.hpp"
#include <arpa/inet.h>
#include <fcntl.h>
#include <errno.h>
#include <endian.h>
#include "log.hpp"
#include "exception.hpp"
#include "singletons/epoll_daemon.hpp"
#include "tcp_session_base.hpp"
using namespace Poseidon;

TcpServerBase::TcpServerBase(const std::string &bindAddr, unsigned bindPort){
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
		u.sin.sin_port = be16toh(bindPort);
		salen = sizeof(::sockaddr_in);
		text = ::inet_ntop(AF_INET, &u.sin.sin_addr, &m_bindAddr[0], m_bindAddr.size());
	} else if(::inet_pton(AF_INET6, bindAddr.c_str(), &u.sin6.sin6_addr) == 1){
		u.sin6.sin6_family = AF_INET6;
		u.sin6.sin6_port = be16toh(bindPort);
		salen = sizeof(::sockaddr_in6);
		text = ::inet_ntop(AF_INET6, &u.sin6.sin6_addr, &m_bindAddr[0], m_bindAddr.size());
	} else {
		DEBUG_THROW(Exception, "Unknown address format: " + bindAddr);
	}
	if(!text){
		DEBUG_THROW(SystemError, errno);
	}
	m_bindAddr.resize(std::strlen(text));
	m_bindAddr += ':';
	m_bindAddr += boost::lexical_cast<std::string>(bindPort);

	LOG_INFO("Creating socket server on ", m_bindAddr, "...");

	m_listen.reset(::socket(u.sa.sa_family, SOCK_STREAM, IPPROTO_TCP));
	if(!m_listen){
		const int code = errno;
		LOG_ERROR("Error creating socket.");
		DEBUG_THROW(SystemError, code);
	}
	const int TRUE_VALUE = true;
	if(::setsockopt(m_listen.get(), SOL_SOCKET, SO_REUSEADDR, &TRUE_VALUE, sizeof(TRUE_VALUE)) != 0){
		const int code = errno;
		LOG_ERROR("Could not set socket to reuse address.");
		DEBUG_THROW(SystemError, code);
	}
	const int flags = ::fcntl(m_listen.get(), F_GETFL);
	if(flags == -1){
		const int code = errno;
		LOG_ERROR("Could not get fcntl flags on socket.");
		DEBUG_THROW(SystemError, code);
	}
	if(::fcntl(m_listen.get(), F_SETFL, flags | O_NONBLOCK) != 0){
		const int code = errno;
		LOG_ERROR("Could not set fcntl flags on socket.");
		DEBUG_THROW(SystemError, code);
	}
	if(::bind(m_listen.get(), &u.sa, salen)){
		const int code = errno;
		LOG_ERROR("Could not bind socket onto specified address.");
		DEBUG_THROW(SystemError, code);
	}
	if(::listen(m_listen.get(), SOMAXCONN)){
		const int code = errno;
		LOG_ERROR("Could not listen on socket.");
		DEBUG_THROW(SystemError, code);
	}
}
TcpServerBase::~TcpServerBase(){
	LOG_INFO("Destroyed socket server on ", m_bindAddr);
}

boost::shared_ptr<TcpSessionBase> TcpServerBase::tryAccept() const {
	ScopedFile client(::accept(m_listen.get(), NULLPTR, NULLPTR));
	if(!client){
		return NULLPTR;
	}
	AUTO(session, onClientConnect(STD_MOVE(client)));
	if(!session){
		return NULLPTR;
	}
	LOG_INFO("Client '", session->getRemoteIp(), "' has connected.");
	return session;
}
