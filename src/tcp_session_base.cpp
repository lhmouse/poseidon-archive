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
#include <netinet/in.h>
#include <netinet/tcp.h>
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

TcpSessionBase::DelayedShutdownGuard::DelayedShutdownGuard(boost::shared_ptr<TcpSessionBase> session)
	: m_session(STD_MOVE(session))
{
	atomicAdd(m_session->m_delayedShutdownGuardCount, 1, ATOMIC_RELAXED);
}
TcpSessionBase::DelayedShutdownGuard::~DelayedShutdownGuard(){
	if(atomicSub(m_session->m_delayedShutdownGuardCount, 1, ATOMIC_RELAXED) != 0){
		return;
	}
	if(atomicLoad(m_session->m_shutdownWrite, ATOMIC_CONSUME)){
	    atomicStore(m_session->m_reallyShutdownWrite, true, ATOMIC_RELEASE);
	    m_session->notifyEpollWriteable();
	}
}

void TcpSessionBase::shutdownTimerProc(const boost::weak_ptr<TcpSessionBase> &weak){
	const AUTO(session, weak.lock());
	if(!session){
		return;
	}

	std::size_t sendBufferSize;
	{
		Mutex::UniqueLock lock;
		sendBufferSize = session->getSendBufferSize(lock);
	}
	if(sendBufferSize != 0){
		LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_INFO, "Send buffer is not empty. Retry later...");
		return;
	}

	try {
		LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_DEBUG, "Connection timed out: remote = ", session->getRemoteInfo());
	} catch(...){
		LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_DEBUG, "Connection timed out: remote is not connected");
	}
	atomicStore(session->m_timedOut, true, ATOMIC_RELEASE);
	session->forceShutdown();
}

TcpSessionBase::TcpSessionBase(UniqueFile socket)
	: m_socket(STD_MOVE(socket)), m_createdTime(getFastMonoClock())
	, m_peerInfo()
	, m_connected(false)
	, m_shutdownRead(false), m_shutdownWrite(false), m_reallyShutdownWrite(false), m_timedOut(false)
	, m_delayedShutdownGuardCount(0)
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

void TcpSessionBase::setConnected(){
	const bool old = atomicExchange(m_connected, true, ATOMIC_ACQ_REL);
	if(!old){
		onConnect();
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
	const Mutex::UniqueLock lock(m_peerInfo.mutex);
	if(m_peerInfo.remote.port == 0){
		m_peerInfo.remote = getRemoteIpPortFromFd(m_socket.get());
		LOG_POSEIDON_DEBUG("TCP session remote info = ", m_peerInfo.remote);
	}
	if(m_peerInfo.local.port == 0){
		m_peerInfo.local = getLocalIpPortFromFd(m_socket.get());
		LOG_POSEIDON_DEBUG("TCP session local info = ", m_peerInfo.local);
	}
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
		fetchPeerInfo();

		const AUTO(bytes, static_cast<std::size_t>(ret.bytesTransferred));
		LOG_POSEIDON_TRACE("Read ", bytes, " byte(s) from ", getRemoteInfo(), ", hex = ", HexDumper(hint, bytes));

		onReadAvail(StreamBuffer(hint, bytes));
	}

	return ret;
}
TcpSessionBase::SyncIoResult TcpSessionBase::syncWrite(void *hint, unsigned long hintSize){
	PROFILE_ME;

	std::size_t bytesAvail;
	{
		const Mutex::UniqueLock lock(m_bufferMutex);
		bytesAvail = m_sendBuffer.peek(hint, hintSize);
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
			fetchPeerInfo();

			const AUTO(bytes, static_cast<std::size_t>(ret.bytesTransferred));
			LOG_POSEIDON_TRACE("Wrote ", bytes, " byte(s) to ", getRemoteInfo(), ", hex = ", HexDumper(hint, bytes));

			const Mutex::UniqueLock lock(m_bufferMutex);
			m_sendBuffer.discard(bytes);
			bytesAvail = m_sendBuffer.size();
		}
	}

	if((bytesAvail == 0) && atomicLoad(m_reallyShutdownWrite, ATOMIC_CONSUME)){
		::shutdown(m_socket.get(), SHUT_WR);
	}
	return ret;
}
std::size_t TcpSessionBase::getSendBufferSize(Mutex::UniqueLock &lock) const {
	Mutex::UniqueLock(m_bufferMutex).swap(lock);
	return m_sendBuffer.size();
}

void TcpSessionBase::onConnect(){
}
void TcpSessionBase::onReadHup() NOEXCEPT {
	shutdownWrite();
}
void TcpSessionBase::onClose(int errCode) NOEXCEPT {
	(void)errCode;

	m_connected = false;
}

bool TcpSessionBase::send(StreamBuffer buffer){
	PROFILE_ME;

	if(atomicLoad(m_reallyShutdownWrite, ATOMIC_CONSUME)){
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
	return atomicLoad(m_shutdownRead, ATOMIC_CONSUME);
}
bool TcpSessionBase::shutdownRead() NOEXCEPT {
	PROFILE_ME;

	bool ret = !atomicLoad(m_shutdownRead, ATOMIC_CONSUME);
	if(ret){
		ret = !atomicExchange(m_shutdownRead, true, ATOMIC_ACQ_REL);
		::shutdown(m_socket.get(), SHUT_RD);
	}
	return ret;
}
bool TcpSessionBase::hasBeenShutdownWrite() const NOEXCEPT {
	return atomicLoad(m_shutdownWrite, ATOMIC_CONSUME);
}
bool TcpSessionBase::shutdownWrite() NOEXCEPT {
	PROFILE_ME;

	bool ret = !atomicLoad(m_shutdownWrite, ATOMIC_CONSUME);
	if(ret){
		ret = !atomicExchange(m_shutdownWrite, true, ATOMIC_ACQ_REL);
		const DelayedShutdownGuard guard(virtualSharedFromThis<TcpSessionBase>());
	}
	return ret;
}
void TcpSessionBase::forceShutdown() NOEXCEPT {
	PROFILE_ME;

	atomicStore(m_shutdownRead, true, ATOMIC_RELEASE);
	atomicStore(m_shutdownWrite, true, ATOMIC_RELEASE);

	::linger lng;
	lng.l_onoff = 1;
	lng.l_linger = 0;
	::setsockopt(m_socket.get(), SOL_SOCKET, SO_LINGER, &lng, sizeof(lng));

	::shutdown(m_socket.get(), SHUT_RDWR);
}

const IpPort &TcpSessionBase::getRemoteInfo() const {
	PROFILE_ME;

	fetchPeerInfo();
	return m_peerInfo.remote;
}
const IpPort &TcpSessionBase::getLocalInfo() const {
	PROFILE_ME;

	fetchPeerInfo();
	return m_peerInfo.local;
}

void TcpSessionBase::setTimeout(boost::uint64_t timeout){
	PROFILE_ME;

	const Mutex::UniqueLock lock(m_timerMutex);
	if(timeout == 0){
		m_shutdownTimer.reset();
	} else {
		if(m_shutdownTimer){
			TimerDaemon::setTime(m_shutdownTimer, timeout, 10000);
		} else {
			m_shutdownTimer = TimerDaemon::registerTimer(timeout, 10000,
				boost::bind(&shutdownTimerProc, virtualWeakFromThis<TcpSessionBase>()), true);
		}
	}
}

void TcpSessionBase::setNoDelay(bool enabled){
	PROFILE_ME;

	const int val = enabled;
	if(::setsockopt(m_socket.get(), IPPROTO_TCP, TCP_NODELAY, &val, sizeof(val)) != 0){
		const int errCode = errno;
		LOG_POSEIDON_WARNING("Error setting TCP socket option: errCode = ", errCode);
		DEBUG_THROW(SystemException, errCode);
	}
}

}
