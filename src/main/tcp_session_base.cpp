#include "precompiled.hpp"
#include "tcp_session_base.hpp"
#define POSEIDON_TCP_SESSION_SSL_IMPL_
#include "tcp_session_ssl_impl.hpp"
#include <arpa/inet.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <fcntl.h>
#include <errno.h>
#include "singletons/epoll_daemon.hpp"
#include "log.hpp"
#include "atomic.hpp"
#include "endian.hpp"
#include "exception.hpp"
#include "utilities.hpp"
using namespace Poseidon;

namespace {

union SockAddr {
	::sockaddr sa;
	::sockaddr_in sin;
	::sockaddr_in6 sin6;
};

ScopedFile setNonBlock(Move<ScopedFile> peer){
	ScopedFile socket(STD_MOVE(peer));
	const int flags = ::fcntl(socket.get(), F_GETFL);
	if(flags == -1){
		const int code = errno;
		LOG_ERROR("Could not get fcntl flags on socket.");
		DEBUG_THROW(SystemError, code);
	}
	if(::fcntl(socket.get(), F_SETFL, flags | O_NONBLOCK) != 0){
		const int code = errno;
		LOG_ERROR("Could not set fcntl flags on socket.");
		DEBUG_THROW(SystemError, code);
	}
	return socket;
}

std::pair<SharedNtmbs, unsigned> getAddrPortFromSockAddr(const SockAddr &sa){
	char ip[64];
	unsigned port;
	const char *ret;
	if(sa.sa.sa_family == AF_INET){
		ret = ::inet_ntop(AF_INET, &sa.sin.sin_addr, ip, sizeof(ip));
		port = loadBe(sa.sin.sin_port);
	} else if(sa.sa.sa_family == AF_INET6){
		ret = ::inet_ntop(AF_INET6, &sa.sin6.sin6_addr, ip, sizeof(ip));
		port = loadBe(sa.sin6.sin6_port);
	} else {
		LOG_WARN("Unknown IP protocol ", sa.sa.sa_family);
		DEBUG_THROW(Exception, "Unknown IP protocol");
	}
	if(!ret){
		LOG_WARN("Failed to format IP address to string.");
		DEBUG_THROW(SystemError);
	}
	return std::make_pair(SharedNtmbs(ip, true), port);
}

std::pair<SharedNtmbs, unsigned> getRemoteAddrFromFd(int fd){
	SockAddr sa;
	::socklen_t salen = sizeof(sa);
	if(::getpeername(fd, &sa.sa, &salen) != 0){
		LOG_ERROR("Failed to get remote socket addr.");
		DEBUG_THROW(SystemError);
	}
	return getAddrPortFromSockAddr(sa);
}
std::pair<SharedNtmbs, unsigned> getLocalAddrFromFd(int fd){
	SockAddr sa;
	::socklen_t salen = sizeof(sa);
	if(::getsockname(fd, &sa.sa, &salen) != 0){
		LOG_ERROR("Failed to get local socket addr.");
		DEBUG_THROW(SystemError);
	}
	return getAddrPortFromSockAddr(sa);
}

}

TcpSessionBase::TcpSessionBase(Move<ScopedFile> socket)
	: m_socket(setNonBlock(STD_MOVE(socket)))
	, m_createdTime(getMonoClock())
	, m_remoteAddr(getRemoteAddrFromFd(m_socket.get()))
	, m_localAddr(getLocalAddrFromFd(m_socket.get()))
	, m_shutdown(false)
{
	LOG_INFO("Created TCP peer to ", m_remoteAddr.first, ':', m_remoteAddr.second);
}
TcpSessionBase::~TcpSessionBase(){
	LOG_INFO("Destroyed TCP peer to ", m_remoteAddr.first, ':', m_remoteAddr.second);
}

bool TcpSessionBase::send(StreamBuffer buffer, bool final){
	bool closed;
	if(final){
		closed = atomicExchange(m_shutdown, true);
	} else {
		closed = atomicLoad(m_shutdown);
	}
	if(closed){
		LOG_DEBUG("Socket to ", m_remoteAddr.first, ':', m_remoteAddr.second, " has been closed.");
		return false;
	}
	if(!buffer.empty()){
		const boost::mutex::scoped_lock lock(m_bufferMutex);
		m_sendBuffer.splice(buffer);
	}
	if(final){
		::shutdown(m_socket.get(), SHUT_RD);
	}
	EpollDaemon::touchSession(virtualSharedFromThis<TcpSessionBase>());
	return true;
}

bool TcpSessionBase::hasBeenShutdown() const {
	return atomicLoad(m_shutdown);
}
bool TcpSessionBase::forceShutdown(){
	const bool ret = !atomicExchange(m_shutdown, true);
	::shutdown(m_socket.get(), SHUT_RDWR);
	return ret;
}

void TcpSessionBase::initSsl(Move<boost::scoped_ptr<SslImpl> > ssl){
	swap(m_ssl, ssl);
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
	::ssize_t ret;
	if(m_ssl){
		ret = m_ssl->doWrite(hint, size);
	} else {
		ret = ::send(m_socket.get(), hint, size, MSG_NOSIGNAL);
	}

	lock.lock();
	if(ret > 0){
		m_sendBuffer.discard(ret);
	}
	return ret;
}
