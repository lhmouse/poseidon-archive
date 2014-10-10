#include "../precompiled.hpp"
#include "tcp_session_base.hpp"
#include <arpa/inet.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <fcntl.h>
#include <errno.h>
#include "exception.hpp"
#include "singletons/epoll_daemon.hpp"
#include "log.hpp"
#include "atomic.hpp"
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
		DEBUG_THROW(Exception, "Unknown IP protocol: "
			+ boost::lexical_cast<std::string>(u.sa.sa_family));
	}
	if(!text){
		DEBUG_THROW(SystemError, errno);
	}
	ret.resize(std::strlen(text));

	return ret;
}

}

TcpSessionBase::TcpSessionBase(Move<ScopedFile> socket)
	: m_socket(STD_MOVE(socket)), m_remoteIp(getIpFromSocket(m_socket.get()))
	, m_shutdown(false)
{
	LOG_INFO("Created TCP peer, remote ip = ", m_remoteIp);
}
TcpSessionBase::~TcpSessionBase(){
	LOG_INFO("Destroyed TCP peer, remote ip = ", m_remoteIp);
}

bool TcpSessionBase::send(StreamBuffer buffer){
	if(atomicLoad(m_shutdown)){
		LOG_DEBUG("Attempting to send data on a closed socket.");
		return false;
	}
	{
		const boost::mutex::scoped_lock lock(m_bufferMutex);
		m_sendBuffer.splice(buffer);
	}
	EpollDaemon::refreshSession(virtualSharedFromThis<TcpSessionBase>());
	return true;
}
bool TcpSessionBase::hasBeenShutdown() const {
	return atomicLoad(m_shutdown);
}
bool TcpSessionBase::shutdown(){
	const bool ret = !atomicExchange(m_shutdown, true);
	::shutdown(getFd(), SHUT_RD);
	return ret;
}
bool TcpSessionBase::forceShutdown(){
	const bool ret = !atomicExchange(m_shutdown, true);
	::shutdown(getFd(), SHUT_RDWR);
	return ret;
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
