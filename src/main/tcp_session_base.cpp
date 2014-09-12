#include "../precompiled.hpp"
#include "tcp_session_base.hpp"
#include "exception.hpp"
#include "singletons/epoll_daemon.hpp"
#include "log.hpp"
#include <arpa/inet.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <fcntl.h>
#include <errno.h>
using namespace Poseidon;

namespace {

std::string getIpFromSocket(int fd){
	std::string ret;

	const int flags = ::fcntl(fd, F_GETFL);
	if(flags == -1){
		DEBUG_THROW(SystemError, errno);
	}
	if(::fcntl(fd, F_SETFL, flags | O_NONBLOCK) != 0){
		DEBUG_THROW(SystemError, errno);
	}

	union {
		::sockaddr sa;
		::sockaddr_in sin;
		::sockaddr_in6 sin6;
	} u;
	::socklen_t salen = sizeof(u);
	if(::getpeername(fd, &u.sa, &salen) != 0){
		DEBUG_THROW(SystemError, errno);
	}
	ret.resize(63);
	const char *text;
	if(u.sa.sa_family == AF_INET){
		text = ::inet_ntop(AF_INET, &u.sin.sin_addr, &ret[0], ret.size());
	} else if(u.sa.sa_family == AF_INET6){
		text = ::inet_ntop(AF_INET6, &u.sin6.sin6_addr, &ret[0], ret.size());
	} else {
		DEBUG_THROW(Exception, "Unknown IP protocol.");
	}
	if(!text){
		DEBUG_THROW(SystemError, errno);
	}
	ret.resize(std::strlen(text));

	return STD_MOVE(ret);
}

}

TcpSessionBase::TcpSessionBase(ScopedFile::Move socket)
	: m_socket(socket), m_remoteIp(getIpFromSocket(m_socket.get()))
	, m_readShutdown(false), m_shutdown(false)
{

	LOG_INFO("Created tcp peer, remote ip = ", m_remoteIp);
}
TcpSessionBase::~TcpSessionBase(){
	if(m_socket){
		LOG_INFO("Destroyed tcp peer, remote ip = ", m_remoteIp);
	}
}

std::size_t TcpSessionBase::peekWriteAvail(boost::mutex::scoped_lock &lock,
	void *data, std::size_t size) const
{
	boost::mutex::scoped_lock(m_bufferMutex).swap(lock);
	if(size == 0){
		return m_sendBuffer.size();
	} else {
		return m_sendBuffer.peek(data, size);
	}
}
void TcpSessionBase::notifyWritten(std::size_t size){
	const boost::mutex::scoped_lock lock(m_bufferMutex);
	m_sendBuffer.discard(size);
}

void TcpSessionBase::send(const void *data, std::size_t size){
	if(atomicLoad(m_shutdown)){
		LOG_DEBUG("Attempting to send data on a closed socket.");
		return;
	}
	StreamBuffer tmp;
	tmp.put(data, size);
	sendUsingMove(tmp);
}
void TcpSessionBase::send(const StreamBuffer &buffer){
	if(atomicLoad(m_shutdown)){
		LOG_DEBUG("Attempting to send data on a closed socket.");
		return;
	}
	StreamBuffer tmp(buffer);
	sendUsingMove(tmp);
}
void TcpSessionBase::sendUsingMove(StreamBuffer &buffer){
	if(atomicLoad(m_shutdown)){
		LOG_DEBUG("Attempting to send data on a closed socket.");
		return;
	}
	{
		const boost::mutex::scoped_lock lock(m_bufferMutex);
		m_sendBuffer.splice(buffer);
	}
	EpollDaemon::refreshSession(virtualSharedFromThis<TcpSessionBase>());
}

void TcpSessionBase::shutdownRead(){
	atomicStore(m_readShutdown, true);
	::shutdown(getFd(), SHUT_RD);
}
void TcpSessionBase::shutdown(){
	atomicStore(m_readShutdown, true);
	atomicStore(m_shutdown, true);
	::shutdown(getFd(), SHUT_RD);
}
void TcpSessionBase::forceShutdown(){
	atomicStore(m_readShutdown, true);
	atomicStore(m_shutdown, true);
	::shutdown(getFd(), SHUT_RDWR);
}
