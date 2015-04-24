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
#include "string.hpp"
#include "time.hpp"
#include "system_exception.hpp"
#include "epoll.hpp"
#include "job_base.hpp"
#include "async_job.hpp"

namespace Poseidon {

namespace {
	void shutdownIfTimeout(const boost::weak_ptr<TcpSessionBase> &weak){
		const AUTO(session, weak.lock());
		if(!session){
			return;
		}
		try {
			LOG_POSEIDON_WARNING("Connection timed out: remote = ", session->getRemoteInfo());
		} catch(...){
			LOG_POSEIDON_WARNING("Connection timed out but the session has not been established.");
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
	boost::weak_ptr<const void> getCategory() const OVERRIDE {
		return m_session;
	}
	void perform() const OVERRIDE {
		m_session->pumpOnClose();
	}
};

TcpSessionBase::TcpSessionBase(UniqueFile socket)
	: m_socket(STD_MOVE(socket)), m_createdTime(getFastMonoClock())
	, m_peerInfo()
	, m_shutdown(false), m_preservedOnReadHup(false)
{
	const int flags = ::fcntl(m_socket.get(), F_GETFL);
	if(flags == -1){
		const int code = errno;
		LOG_POSEIDON_ERROR("Could not get fcntl flags on socket.");
		DEBUG_THROW(SystemException, code);
	}
	if(::fcntl(m_socket.get(), F_SETFL, flags | O_NONBLOCK) != 0){
		const int code = errno;
		LOG_POSEIDON_ERROR("Could not set fcntl flags on socket.");
		DEBUG_THROW(SystemException, code);
	}
}
TcpSessionBase::~TcpSessionBase(){
	try {
		LOG_POSEIDON_INFO("Destroying TCP session: remote = ", getRemoteInfo());
	} catch(...){
		LOG_POSEIDON_INFO("Destroying TCP session that has not been established.");
	}

	pumpOnClose();
}

void TcpSessionBase::initSsl(Move<boost::scoped_ptr<SslFilterBase> > sslFilter){
	swap(m_sslFilter, sslFilter);
}

void TcpSessionBase::setEpoll(boost::weak_ptr<const boost::weak_ptr<Epoll> > epoll) NOEXCEPT {
	const boost::mutex::scoped_lock lock(m_bufferMutex);
	const AUTO(oldPtr, m_epoll.lock());
	if(oldPtr){
		const AUTO(old, oldPtr->lock());
		if(old){
			old->notifyUnlinked(this);
		}
	}
	m_epoll = STD_MOVE(epoll);
}

void TcpSessionBase::pumpOnClose() NOEXCEPT {
	const boost::mutex::scoped_lock lock(m_onCloseMutex);
	while(!m_onCloseQueue.empty()){
		try {
			enqueueAsyncJob(STD_MOVE(m_onCloseQueue.back()));
		} catch(std::exception &e){
			LOG_POSEIDON_ERROR("std::exception thrown while enqueueing onClose callback: what = ", e.what());
		} catch(...){
			LOG_POSEIDON_ERROR("Unknown exception thrown while enqueueing onClose callback.");
		}
		m_onCloseQueue.pop_back();
	}
}
void TcpSessionBase::onClose() NOEXCEPT {
	try {
		// 不要在这个地方检查队列是否为空，因为这里是 epoll 线程，
		// 而主线程有可能在这个事件之后加入了一些回调，那样它们就不会被调用。
		enqueueJob(boost::make_shared<OnCloseJob>(virtualSharedFromThis<TcpSessionBase>()));
	} catch(std::exception &e){
		LOG_POSEIDON_ERROR("std::exception thrown while enqueueing onClose job: what = ", e.what());
	} catch(...){
		LOG_POSEIDON_ERROR("Unknown exception thrown while enqueueing onClose job");
	}
}
void TcpSessionBase::onReadHup() NOEXCEPT {
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
	const boost::mutex::scoped_lock lock(m_bufferMutex);
	if(!buffer.empty()){
		m_sendBuffer.splice(buffer);
	}
	const AUTO(ptr, m_epoll.lock());
	if(ptr){
		const AUTO(epoll, ptr->lock());
		if(epoll){
			epoll->notifyWriteable(this);
		}
	}
	if(fin){
		atomicExchange(m_preservedOnReadHup, false, ATOMIC_ACQ_REL);
		shutdown();
	}
	return true;
}

bool TcpSessionBase::hasBeenShutdown() const {
	return atomicLoad(m_shutdown, ATOMIC_ACQUIRE);
}
bool TcpSessionBase::shutdown() NOEXCEPT {
	const bool ret = !atomicExchange(m_shutdown, true, ATOMIC_ACQ_REL);
	::shutdown(m_socket.get(), SHUT_RD);
	return ret;
}
bool TcpSessionBase::forceShutdown() NOEXCEPT {
	const bool ret = !atomicExchange(m_shutdown, true, ATOMIC_ACQ_REL);
	::shutdown(m_socket.get(), SHUT_RDWR);
	return ret;
}

bool TcpSessionBase::isPreservedOnReadHup() const NOEXCEPT {
	return atomicLoad(m_preservedOnReadHup, ATOMIC_ACQUIRE);
}
bool TcpSessionBase::setPreservedOnReadHup(bool value) NOEXCEPT {
	if(!value && hasBeenShutdown()){
		shutdown(); // noexcept
	}
	return atomicExchange(m_preservedOnReadHup, value, ATOMIC_ACQ_REL);
}

long TcpSessionBase::syncReadAndProcess(void *hint, unsigned long hintSize){
	::ssize_t ret;
	if(m_sslFilter){
		ret = m_sslFilter->read(hint, hintSize);
	} else {
		ret = ::recv(m_socket.get(), hint, hintSize, MSG_NOSIGNAL);
	}
	if(ret > 0){
		LOG_POSEIDON_TRACE("Read ", ret, " byte(s) from ", getRemoteInfo(),
			", hex = ", HexDumper(hint, static_cast<std::size_t>(ret)));
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
void TcpSessionBase::setTimeout(boost::uint64_t timeout){
	boost::shared_ptr<const TimerItem> shutdownTimer;
	if(timeout != 0){
		shutdownTimer = TimerDaemon::registerTimer(
			timeout, 0, boost::bind(&shutdownIfTimeout, virtualWeakFromThis<TcpSessionBase>()));
	}
	{
		const boost::mutex::scoped_lock lock(m_timerMutex);
		m_shutdownTimer.swap(shutdownTimer);
	}
}

}
