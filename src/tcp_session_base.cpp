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
#include "profiler.hpp"

namespace Poseidon {

namespace {
	void shutdownIfTimeout(const boost::weak_ptr<TcpSessionBase> &weak){
		const AUTO(session, weak.lock());
		if(!session){
			return;
		}

		try {
			LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_INFO, "Connection timed out: remote = ", session->getRemoteInfo());
		} catch(...){
			LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_INFO, "Connection timed out but the session has not been established.");
		}
		session->forceShutdown();
	}
}

TcpSessionBase::DelayedShutdownGuard::DelayedShutdownGuard(boost::shared_ptr<TcpSessionBase> session)
	: m_session(STD_MOVE(session))
{
	atomicAdd(m_session->m_delayedShutdownGuardCount, 1, ATOMIC_RELAXED);
}
TcpSessionBase::DelayedShutdownGuard::~DelayedShutdownGuard(){
	if(atomicSub(m_session->m_delayedShutdownGuardCount, 1, ATOMIC_RELAXED) == 0){
		m_session->shutdownWrite();
	}
}

TcpSessionBase::TcpSessionBase(UniqueFile socket)
	: m_socket(STD_MOVE(socket)), m_createdTime(getFastMonoClock())
	, m_peerInfo()
	, m_shutdownRead(false), m_shutdownWrite(false), m_delayedShutdownGuardCount(0), m_reallyShutdownWrite(false)
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
}

void TcpSessionBase::initSsl(Move<boost::scoped_ptr<SslFilterBase> > sslFilter){
	swap(m_sslFilter, sslFilter);
}

void TcpSessionBase::setEpoll(boost::weak_ptr<const boost::weak_ptr<Epoll> > epoll) NOEXCEPT {
	const Mutex::UniqueLock lock(m_bufferMutex);
	const AUTO(oldPtr, m_epoll.lock());
	if(oldPtr){
		const AUTO(old, oldPtr->lock());
		if(old){
			old->notifyUnlinked(this);
		}
	}
	m_epoll = STD_MOVE(epoll);
}
void TcpSessionBase::notifyEpollWriteable() NOEXCEPT {
	const AUTO(ptr, m_epoll.lock());
	if(ptr){
		const AUTO(epoll, ptr->lock());
		if(epoll){
			epoll->notifyWriteable(this);
		}
	}
}

void TcpSessionBase::fetchPeerInfo() const {
	if(atomicLoad(m_peerInfo.fetched, ATOMIC_ACQUIRE)){
		return;
	}
	{
		const Mutex::UniqueLock lock(m_peerInfo.mutex);
		if(atomicLoad(m_peerInfo.fetched, ATOMIC_ACQUIRE)){
			return;
		}
		m_peerInfo.remote = getRemoteIpPortFromFd(m_socket.get());
		m_peerInfo.local = getLocalIpPortFromFd(m_socket.get());
		atomicStore(m_peerInfo.fetched, true, ATOMIC_RELEASE);
	}
	LOG_POSEIDON_INFO("TCP session: remote = ", m_peerInfo.remote, ", local = ", m_peerInfo.local);
}

TcpSessionBase::SyncIoResult TcpSessionBase::syncReadAndProcess(void *hint, unsigned long hintSize){
	PROFILE_ME;

	SyncIoResult ret;
	if(m_sslFilter){
		ret.bytesTransferred = m_sslFilter->read(hint, hintSize);
	} else {
		ret.bytesTransferred = ::recv(m_socket.get(), hint, hintSize, MSG_NOSIGNAL);
	}
	ret.errCode = errno;

	if(ret.bytesTransferred > 0){
		const AUTO(bytes, static_cast<std::size_t>(ret.bytesTransferred));
		LOG_POSEIDON_TRACE("Read ", bytes, " byte(s) from ", getRemoteInfo(), ", hex = ", HexDumper(hint, bytes));

		onReadAvail(hint, bytes);
	}

	return ret;
}
TcpSessionBase::SyncIoResult TcpSessionBase::syncWrite(void *hint, unsigned long hintSize){
	PROFILE_ME;

	std::size_t bytesAvail;
	bool empty;
	{
		const Mutex::UniqueLock lock(m_bufferMutex);
		bytesAvail = m_sendBuffer.peek(hint, hintSize);
		empty = m_sendBuffer.empty();
	}

	SyncIoResult ret;
	if(bytesAvail == 0){
		ret.bytesTransferred = 0;
	} else {
		if(m_sslFilter){
			ret.bytesTransferred = m_sslFilter->write(hint, bytesAvail);
		} else {
			ret.bytesTransferred = ::send(m_socket.get(), hint, bytesAvail, MSG_NOSIGNAL);
		}
		ret.errCode = errno;

		if(ret.bytesTransferred > 0){
			const AUTO(bytes, static_cast<std::size_t>(ret.bytesTransferred));
			LOG_POSEIDON_TRACE("Wrote ", bytes, " byte(s) to ", getRemoteInfo(), ", hex = ", HexDumper(hint, bytes));

			const Mutex::UniqueLock lock(m_bufferMutex);
			m_sendBuffer.discard(bytes);
			empty = m_sendBuffer.empty();
		}
	}

	if(empty && atomicLoad(m_reallyShutdownWrite, ATOMIC_ACQUIRE)){
		::shutdown(m_socket.get(), SHUT_WR);
	}
	return ret;
}
bool TcpSessionBase::isSendBufferEmpty(Mutex::UniqueLock &lock) const {
	Mutex::UniqueLock(m_bufferMutex).swap(lock);
	return m_sendBuffer.empty();
}

void TcpSessionBase::onReadHup() NOEXCEPT {
	shutdownWrite();
}
void TcpSessionBase::onWriteHup() NOEXCEPT {
}
void TcpSessionBase::onClose(int errCode) NOEXCEPT {
	(void)errCode;
}

bool TcpSessionBase::send(StreamBuffer buffer){
	if(atomicLoad(m_reallyShutdownWrite, ATOMIC_ACQUIRE)){
		LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_DEBUG,
			"Connection has been shut down for writing: remote = ", getRemoteInfo());
		return false;
	}

	const Mutex::UniqueLock lock(m_bufferMutex);
	if(!buffer.empty()){
		m_sendBuffer.splice(buffer);
	}
	notifyEpollWriteable();
	return true;
}

bool TcpSessionBase::hasBeenShutdownRead() const NOEXCEPT {
	return atomicLoad(m_shutdownRead, ATOMIC_ACQUIRE);
}
bool TcpSessionBase::shutdownRead() NOEXCEPT {
	const bool ret = !atomicExchange(m_shutdownRead, true, ATOMIC_ACQ_REL);
	::shutdown(m_socket.get(), SHUT_RD);
	return ret;
}
bool TcpSessionBase::hasBeenShutdownWrite() const NOEXCEPT {
	return atomicLoad(m_shutdownWrite, ATOMIC_ACQUIRE);
}
bool TcpSessionBase::shutdownWrite() NOEXCEPT {
	const bool ret = !atomicExchange(m_shutdownWrite, true, ATOMIC_ACQ_REL);
	if(atomicLoad(m_delayedShutdownGuardCount, ATOMIC_RELAXED) == 0){
		atomicStore(m_reallyShutdownWrite, true, ATOMIC_RELEASE);
		notifyEpollWriteable();
	}
	return ret;
}
void TcpSessionBase::forceShutdown() NOEXCEPT {
	atomicStore(m_shutdownRead, true, ATOMIC_RELEASE);
	atomicStore(m_shutdownWrite, true, ATOMIC_RELEASE);
	::shutdown(m_socket.get(), SHUT_RDWR);
}

const IpPort &TcpSessionBase::getRemoteInfo() const {
	fetchPeerInfo();
	return m_peerInfo.remote;
}
const IpPort &TcpSessionBase::getLocalInfo() const {
	fetchPeerInfo();
	return m_peerInfo.local;
}

void TcpSessionBase::setTimeout(boost::uint64_t timeout){
	if(timeout == 0){
		const Mutex::UniqueLock lock(m_timerMutex);
		m_shutdownTimer.reset();
	} else {
		boost::shared_ptr<TimerItem> timer;
		{
			const Mutex::UniqueLock lock(m_timerMutex);
			timer = m_shutdownTimer;
		}
		if(timer){
			TimerDaemon::setTime(timer, timeout, 0);
		} else {
			timer = TimerDaemon::registerTimer(timeout, 0,
				boost::bind(&shutdownIfTimeout, virtualWeakFromThis<TcpSessionBase>()));

			const Mutex::UniqueLock lock(m_timerMutex);
			m_shutdownTimer = STD_MOVE(timer);
		}
	}
}

}
