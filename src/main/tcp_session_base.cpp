// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2015, LH_Mouse. All wrongs reserved.

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
#include "job_base.hpp"
using namespace Poseidon;

namespace {

void shutdownIfTimeout(boost::weak_ptr<TcpSessionBase> weak){
	const AUTO(session, weak.lock());
	if(!session){
		return;
	}
	try {
		LOG_POSEIDON_WARN("Connection timed out: remote = ", session->getRemoteInfo());
	} catch(...){
		LOG_POSEIDON_WARN("Connection timed out but the session has not been established.");
	}
	session->forceShutdown();
}

}

class TcpSessionBase::OnCloseJob : public JobBase {
private:
	const boost::shared_ptr<TcpSessionBase> m_session; // 强引用。

public:
	explicit OnCloseJob(boost::shared_ptr<TcpSessionBase> session)
		: m_session(STD_MOVE(session))
	{
	}

private:
	void perform() OVERRIDE {
		m_session->pumpOnClose();
	}
};

TcpSessionBase::TcpSessionBase(UniqueFile socket)
	: m_socket(STD_MOVE(socket)), m_createdTime(getFastMonoClock())
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
	try {
		LOG_POSEIDON_INFO("Destroying TCP session:  remote = ", getRemoteInfo());
	} catch(...){
		LOG_POSEIDON_INFO("Destroying TCP session that has not been established.");
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
void TcpSessionBase::pumpOnClose() NOEXCEPT {
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

void TcpSessionBase::onClose() NOEXCEPT {
	try {
		// 不要在这个地方检查队列是否为空，因为这里是 epoll 线程，
		// 而主线程有可能在这个事件之后加入了一些回调，那样它们就不会被调用。
		pendJob(boost::make_shared<OnCloseJob>(virtualSharedFromThis<TcpSessionBase>()));
	} catch(std::exception &e){
		LOG_POSEIDON_ERROR("std::exception thrown while pending onClose job: what = ", e.what());
	} catch(...){
		LOG_POSEIDON_ERROR("Unknown exception thrown while pending onClose job");
	}
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
		LOG_POSEIDON_TRACE("Read ", ret, " byte(s) from ", getRemoteInfo(), ", hex = ",
			HexDumper(hint, static_cast<std::size_t>(ret)));
		onReadAvail(hint, static_cast<std::size_t>(ret));
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
	if(ret > 0){
		LOG_POSEIDON_TRACE("Wrote ", ret, " byte(s) to ", getRemoteInfo(), ", hex = ",
			HexDumper(hint, static_cast<std::size_t>(ret)));
	}

	lock.lock();
	if(ret > 0){
		m_sendBuffer.discard(static_cast<std::size_t>(ret));
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

void TcpSessionBase::registerOnClose(boost::function<void ()> callback){
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
