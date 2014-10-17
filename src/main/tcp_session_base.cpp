#include "../precompiled.hpp"
#include "tcp_session_base.hpp"
#define POSEIDON_TCP_SESSION_SSL_IMPL_
#include "tcp_session_ssl_impl.hpp"
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
	const int flags = ::fcntl(m_socket.get(), F_GETFL);
	if(flags == -1){
		const int code = errno;
		LOG_ERROR("Could not get fcntl flags on socket.");
		DEBUG_THROW(SystemError, code);
	}
	if(::fcntl(m_socket.get(), F_SETFL, flags | O_NONBLOCK) != 0){
		const int code = errno;
		LOG_ERROR("Could not set fcntl flags on socket.");
		DEBUG_THROW(SystemError, code);
	}

	LOG_INFO("Created TCP peer, remote IP = ", m_remoteIp);
}
TcpSessionBase::~TcpSessionBase(){
	LOG_INFO("Destroyed TCP peer, remote IP = ", m_remoteIp);
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
	EpollDaemon::touchSession(virtualSharedFromThis<TcpSessionBase>());
	return true;
}

const std::string &TcpSessionBase::getRemoteIp() const {
	return m_remoteIp;
}
bool TcpSessionBase::hasBeenShutdown() const {
	return atomicLoad(m_shutdown);
}
bool TcpSessionBase::shutdown(){
	const bool ret = !atomicExchange(m_shutdown, true);
	::shutdown(m_socket.get(), SHUT_RD);
	return ret;
}
bool TcpSessionBase::forceShutdown(){
	const bool ret = !atomicExchange(m_shutdown, true);
	::shutdown(m_socket.get(), SHUT_RDWR);
	return ret;
}

void TcpSessionBase::initSsl(Move<boost::scoped_ptr<SslImpl> > ssl){
	ssl.swap(m_ssl);
}

long TcpSessionBase::doRead(void *data, unsigned long size){
	::ssize_t ret;
	if(m_ssl){
		ret = m_ssl->doRead(data, size);
	} else {
		ret = ::recv(m_socket.get(), data, size, MSG_NOSIGNAL);
	}
	if(ret > 0){
		onReadAvail(data, ret);
	}
	return ret;
}
long TcpSessionBase::doWrite(boost::mutex::scoped_lock &lock,
	void *hint, unsigned long hintSize)
{
	boost::mutex::scoped_lock(m_bufferMutex).swap(lock);
	const std::size_t size = m_sendBuffer.peek(hint, hintSize);
	lock.unlock();
	if(size == 0){
		return 0;
	}

	lock.lock();
	::ssize_t ret;
	if(m_ssl){
		ret = m_ssl->doWrite(hint, size);
	} else {
		ret = ::send(m_socket.get(), hint, size, MSG_NOSIGNAL);
	}
	if(ret > 0){
		m_sendBuffer.discard(ret);
	}
	return ret;
}

bool TcpSessionBase::shutdown(StreamBuffer buffer){
	const bool ret = !atomicExchange(m_shutdown, true);
	if(ret){
		const boost::mutex::scoped_lock lock(m_bufferMutex);
		m_sendBuffer.splice(buffer);
	}
	::shutdown(m_socket.get(), SHUT_RD);
	return ret;
}
