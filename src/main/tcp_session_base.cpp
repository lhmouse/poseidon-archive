// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014, LH_Mouse. All wrongs reserved.

#include "precompiled.hpp"
#include "tcp_session_base.hpp"
#include "ssl_filter_base.hpp"
#include "ip_port.hpp"
#include <sys/types.h>
#include <sys/socket.h>
#include <fcntl.h>
#include <errno.h>
#include "singletons/timer_daemon.hpp"
#include "log.hpp"
#include "atomic.hpp"
#include "endian.hpp"
#include "exception.hpp"
#include "utilities.hpp"
#include "epoll.hpp"
using namespace Poseidon;

namespace {

void shutdownIfTimeout(boost::weak_ptr<TcpSessionBase> weak){
	const AUTO(session, weak.lock());
	if(!session){
		return;
	}
	LOG_POSEIDON_WARN("Connection timed out.");
	session->forceShutdown();
}

}

TcpSessionBase::TcpSessionBase(UniqueFile socket)
	: m_socket(STD_MOVE(socket)), m_createdTime(getMonoClock())
	, m_peerInfo(), m_shutdown(false), m_epoll(NULLPTR)
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
	LOG_POSEIDON_INFO(
		"Destroyed TCP session: remote = ", m_peerInfo.remote, ", local = ", m_peerInfo.local);
}

void TcpSessionBase::onClose() NOEXCEPT {
	std::deque<boost::function<void ()> > onCloseQueue;
	{
		const boost::mutex::scoped_lock lock(m_onCloseMutex);
		onCloseQueue.swap(m_onCloseQueue);
	}
	while(!onCloseQueue.empty()){
		try {
			onCloseQueue.back()();
		} catch(std::exception &e){
			LOG_POSEIDON_ERROR("std::exception thrown while in close callback: what = ", e.what());
		} catch(...){
			LOG_POSEIDON_ERROR("Unknown exception thrown while in close callback.");
		}
		onCloseQueue.pop_back();
	}
}

void TcpSessionBase::addMonitor() NOEXCEPT {
	atomicAdd(m_monitorCount, 1, ATOMIC_RELEASE);
}
void TcpSessionBase::removeMonitor() NOEXCEPT {
	if(atomicSub(m_monitorCount, 1, ATOMIC_ACQUIRE) == 0){
		try {
			send(StreamBuffer(), true);
		} catch(...){
			forceShutdown();
		}
	}
}

void TcpSessionBase::setEpoll(Epoll *epoll) NOEXCEPT {
	const boost::mutex::scoped_lock lock(m_bufferMutex);
	assert(!(m_epoll && epoll));
	m_epoll = epoll;
}

void TcpSessionBase::initSsl(Move<boost::scoped_ptr<SslFilterBase> > sslFilter){
	swap(m_sslFilter, sslFilter);
}

void TcpSessionBase::fetchPeerInfo() const {
	if(atomicLoad(m_peerInfo.fetched, ATOMIC_ACQUIRE)){
		return;
	}
	const boost::mutex::scoped_lock lock(m_peerInfo.mutex);
	if(atomicLoad(m_peerInfo.fetched, ATOMIC_ACQUIRE)){
		return;
	}

	m_peerInfo.remote = getRemoteIpPortFromFd(m_socket.get());
	m_peerInfo.local = getLocalIpPortFromFd(m_socket.get());
	LOG_POSEIDON_INFO("TCP session: remote = ", m_peerInfo.remote, ", local = ", m_peerInfo.local);
	atomicStore(m_peerInfo.fetched, true, ATOMIC_RELEASE);
}

bool TcpSessionBase::send(StreamBuffer buffer, bool fin){
	bool closed;
	if(fin){
		closed = atomicExchange(m_shutdown, true, ATOMIC_ACQ_REL);
	} else {
		closed = atomicLoad(m_shutdown, ATOMIC_ACQUIRE);
	}
	if(closed){
		LOG_POSEIDON_DEBUG("Unable to send data because this socket has been closed.");
		return false;
	}

	const boost::mutex::scoped_lock lock(m_bufferMutex);
	if(!buffer.empty()){
		m_sendBuffer.splice(buffer);
	}
	if(fin){
		::shutdown(m_socket.get(), SHUT_RD);
	}
	if(m_epoll){
		m_epoll->notifyWriteable(this);
	}
	return true;
}
bool TcpSessionBase::hasBeenShutdown() const {
	return atomicLoad(m_shutdown, ATOMIC_ACQUIRE);
}
bool TcpSessionBase::forceShutdown() NOEXCEPT {
	const bool ret = !atomicExchange(m_shutdown, true, ATOMIC_ACQ_REL);
	::shutdown(m_socket.get(), SHUT_RDWR);
	return ret;
}

long TcpSessionBase::syncReadAndProcess(void *hint, unsigned long hintSize){
	::ssize_t ret;
	if(m_sslFilter){
		ret = m_sslFilter->read(hint, hintSize);
	} else {
		ret = ::recv(m_socket.get(), hint, hintSize, MSG_NOSIGNAL);
	}
	if(ret > 0){
		onReadAvail(hint, ret);
	}
	return ret;
}
long TcpSessionBase::syncWrite(boost::mutex::scoped_lock &lock, void *hint, unsigned long hintSize){
	boost::mutex::scoped_lock(m_bufferMutex).swap(lock);
	const std::size_t size = m_sendBuffer.peek(hint, hintSize);
	lock.unlock();

	if(size == 0){
		return 0;
	}
	::ssize_t ret;
	if(m_sslFilter){
		ret = m_sslFilter->write(hint, size);
	} else {
		ret = ::send(m_socket.get(), hint, size, MSG_NOSIGNAL);
	}

	lock.lock();
	if(ret > 0){
		m_sendBuffer.discard(ret);
	}
	return ret;
}

void TcpSessionBase::setOnClose(boost::function<void ()> callback){
	const boost::mutex::scoped_lock lock(m_onCloseMutex);
	m_onCloseQueue.push_back(boost::function<void ()>());
	m_onCloseQueue.back().swap(callback);
}
void TcpSessionBase::setTimeout(unsigned long long timeout){
	boost::shared_ptr<const TimerItem> shutdownTimer;
	if(timeout != 0){
		shutdownTimer = TimerDaemon::registerTimer(timeout, 0,
			boost::bind(&shutdownIfTimeout, virtualWeakFromThis<TcpSessionBase>()));
	}
	{
		const boost::mutex::scoped_lock lock(m_timerMutex);
		m_shutdownTimer.swap(shutdownTimer);
	}
}

const IpPort &TcpSessionBase::getRemoteInfo() const {
	fetchPeerInfo();
	return m_peerInfo.remote;
}
const IpPort &TcpSessionBase::getLocalInfo() const {
	fetchPeerInfo();
	return m_peerInfo.local;
}
