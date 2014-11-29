// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014, LH_Mouse. All wrongs reserved.

#include "precompiled.hpp"
#include "tcp_session_base.hpp"
#define POSEIDON_TCP_SESSION_SSL_IMPL_
#include "tcp_session_ssl_impl.hpp"
#define POSEIDON_SOCK_ADDR_
#include "sock_addr.hpp"
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
		LOG_POSEIDON_INFO("Destroyed TCP session: remote = ", m_peerInfo.remote,
			", local = ", m_peerInfo.local);
	} else {
		LOG_POSEIDON_INFO("A TCP session that wasn't fully established has been closed.");
	}
}

bool TcpSessionBase::send(StreamBuffer buffer, bool fin){
	bool closed;
	if(fin){
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
	if(fin){
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

	m_peerInfo.remote = getRemoteIpPortFromFd(m_socket.get());
	m_peerInfo.local = getLocalIpPortFromFd(m_socket.get());
	LOG_POSEIDON_INFO("TCP session: remote = ", m_peerInfo.remote, ", local = ", m_peerInfo.local);
	atomicStore(m_peerInfo.fetched, true);
}

long TcpSessionBase::syncRead(void *data, unsigned long size){
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
long TcpSessionBase::syncWrite(boost::mutex::scoped_lock &lock, void *hint, unsigned long hintSize){
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

const IpPort &TcpSessionBase::getRemoteInfo() const {
	fetchPeerInfo();
	return m_peerInfo.remote;
}
const IpPort &TcpSessionBase::getLocalInfo() const {
	fetchPeerInfo();
	return m_peerInfo.local;
}
