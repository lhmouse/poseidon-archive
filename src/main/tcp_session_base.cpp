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
		LOG_POSEIDON_WARN("Unknown IP protocol ", sa.sa.sa_family);
		DEBUG_THROW(Exception, "Unknown IP protocol");
	}
	if(!ret){
		const int code = errno;
		LOG_POSEIDON_WARN("Failed to format IP address to string.");
		DEBUG_THROW(SystemError, code);
	}
	return std::make_pair(SharedNtmbs(ip, true), port);
}

}

TcpSessionBase::TcpSessionBase(ScopedFile socket)
	: m_socket(STD_MOVE(socket)), m_createdTime(getMonoClock())
	, m_peerInfo(), m_shutdown(false)
{
	const int flags = ::fcntl(m_socket.get(), F_GETFL);
	if(flags == -1){
		const int code = errno;
		LOG_POSEIDON_ERROR("Could not get fcntl flags on socket.");
		DEBUG_THROW(SystemError, code);
	}
	if(::fcntl(m_socket.get(), F_SETFL, flags | O_NONBLOCK) != 0){
		const int code = errno;
		LOG_POSEIDON_ERROR("Could not set fcntl flags on socket.");
		DEBUG_THROW(SystemError, code);
	}
}
TcpSessionBase::~TcpSessionBase(){
	if(atomicLoad(m_peerInfo.fetched)){
		LOG_POSEIDON_INFO("Destroyed TCP session, remote address is ",
			m_peerInfo.remote.first, ':', m_peerInfo.remote.second);
	} else {
		LOG_POSEIDON_INFO("A TCP session that wasn't fully established has been closed.");
	}
}

bool TcpSessionBase::send(StreamBuffer buffer, bool final){
	bool closed;
	if(final){
		closed = atomicExchange(m_shutdown, true);
	} else {
		closed = atomicLoad(m_shutdown);
	}
	if(closed){
		LOG_POSEIDON_DEBUG("Socket has been closed.");
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

void TcpSessionBase::fetchPeerInfo() const {
	if(atomicLoad(m_peerInfo.fetched)){
		return;
	}

	const boost::mutex::scoped_lock lock(m_peerInfo.mutex);
	if(atomicLoad(m_peerInfo.fetched)){
		return;
	}

	SockAddr sa;
	::socklen_t salen;

	salen = sizeof(sa);
	if(::getpeername(m_socket.get(), &sa.sa, &salen) != 0){
		const int code = errno;
		LOG_POSEIDON_ERROR("Failed to get remote socket addr.");
		DEBUG_THROW(SystemError, code);
	}
	m_peerInfo.remote = getAddrPortFromSockAddr(sa);

	salen = sizeof(sa);
	if(::getsockname(m_socket.get(), &sa.sa, &salen) != 0){
		const int code = errno;
		LOG_POSEIDON_ERROR("Failed to get local socket addr.");
		DEBUG_THROW(SystemError, code);
	}
	m_peerInfo.local = getAddrPortFromSockAddr(sa);

	LOG_POSEIDON_INFO("Established TCP session, remote address is ",
		m_peerInfo.remote.first, ':', m_peerInfo.remote.second);

	atomicStore(m_peerInfo.fetched, true);
}

long TcpSessionBase::doRead(void *data, unsigned long size){
	fetchPeerInfo();

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
long TcpSessionBase::doWrite(boost::mutex::scoped_lock &lock, void *hint, unsigned long hintSize){
	fetchPeerInfo();

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

const SharedNtmbs &TcpSessionBase::getRemoteIp() const {
	fetchPeerInfo();
	return m_peerInfo.remote.first;
}
unsigned TcpSessionBase::getRemotePort() const {
	fetchPeerInfo();
	return m_peerInfo.remote.second;
}
const SharedNtmbs &TcpSessionBase::getLocalIp() const {
	fetchPeerInfo();
	return m_peerInfo.local.first;
}
unsigned TcpSessionBase::getLocalPort() const {
	fetchPeerInfo();
	return m_peerInfo.local.second;
}
