// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2015, LH_Mouse. All wrongs reserved.

#include "precompiled.hpp"
#include "epoll.hpp"
#include "tcp_session_base.hpp"
#include <sys/epoll.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <errno.h>
#include "multi_index_map.hpp"
#include "system_exception.hpp"
#include "log.hpp"
#include "time.hpp"
#include "errno.hpp"

namespace Poseidon {

namespace {
	enum {
		MAX_PUMP_COUNT = 256
	};

	struct SessionMapElement {
		boost::shared_ptr<TcpSessionBase> session;
		boost::shared_ptr<const boost::weak_ptr<Epoll> > epoll;

		TcpSessionBase *addr;
		// 时间戳，零表示无数据可读/写。
		boost::uint64_t lastRead;
		boost::uint64_t lastWritten;

		mutable bool readHup;
		mutable bool writeHup;

		SessionMapElement(boost::weak_ptr<Epoll> epoll_, boost::shared_ptr<TcpSessionBase> session_)
			: session(STD_MOVE(session_)), epoll(boost::make_shared<boost::weak_ptr<Epoll> >(STD_MOVE(epoll_)))
			, addr(session.get()), lastRead(0), lastWritten(0)
			, readHup(false), writeHup(false)
		{
		}
	};

	MULTI_INDEX_MAP(SessionMap, SessionMapElement,
		UNIQUE_MEMBER_INDEX(addr)
		MULTI_MEMBER_INDEX(lastRead)
		MULTI_MEMBER_INDEX(lastWritten)
	)

	enum {
		IDX_ADDR,
		IDX_READ,
		IDX_WRITE,
	};
}

struct Epoll::SessionMapDelegator : public SessionMap {
};

Epoll::Epoll(){
	if(!m_epoll.reset(::epoll_create(4096))){
		DEBUG_THROW(SystemException);
	}
	m_sessions.reset(new SessionMapDelegator);
}
Epoll::~Epoll(){
}

void Epoll::notifyWriteable(TcpSessionBase *session) NOEXCEPT {
	const AUTO(now, getFastMonoClock());
	const Mutex::UniqueLock lock(m_mutex);
	const AUTO(it, m_sessions->find<IDX_ADDR>(session));
	if(it == m_sessions->end<IDX_ADDR>()){
		LOG_POSEIDON_WARNING("Session is not in epoll?");
		return;
	}
	m_sessions->setKey<IDX_ADDR, IDX_WRITE>(it, now);
}
void Epoll::notifyUnlinked(TcpSessionBase *session) NOEXCEPT {
	const Mutex::UniqueLock lock(m_mutex);
	const AUTO(it, m_sessions->find<IDX_ADDR>(session));
	if(it == m_sessions->end<IDX_ADDR>()){
		LOG_POSEIDON_WARNING("Session is not in epoll.");
		return;
	}
	if(::epoll_ctl(m_epoll.get(), EPOLL_CTL_DEL, session->getFd(), (::epoll_event *)-1) != 0){
		const int errCode = errno;
		LOG_POSEIDON_WARNING("Error deleting from epoll: errno = ", errCode);
	}
	m_sessions->erase<IDX_ADDR>(it);
}

void Epoll::addSession(const boost::shared_ptr<TcpSessionBase> &session){
	const Mutex::UniqueLock lock(m_mutex);
	const AUTO(result, m_sessions->insert(SessionMapElement(shared_from_this(), session)));
	if(!result.second){
		LOG_POSEIDON_WARNING("Session is already in epoll.");
		return;
	}
	::epoll_event event;
	event.events = static_cast< ::uint32_t>(EPOLLIN | EPOLLOUT | EPOLLRDHUP | EPOLLET);
	event.data.ptr = session.get();
	if(::epoll_ctl(m_epoll.get(), EPOLL_CTL_ADD, session->getFd(), &event) != 0){
		const int errCode = errno;
		m_sessions->erase(result.first); // !!
		DEBUG_THROW(SystemException, errCode);
	}
	session->setEpoll(result.first->epoll);
}
void Epoll::removeSession(const boost::shared_ptr<TcpSessionBase> &session){
	const Mutex::UniqueLock lock(m_mutex);
	const AUTO(it, m_sessions->find<IDX_ADDR>(session.get()));
	if(it == m_sessions->end<IDX_ADDR>()){
		LOG_POSEIDON_WARNING("Session is not in epoll.");
		return;
	}
	if(::epoll_ctl(m_epoll.get(), EPOLL_CTL_DEL, session->getFd(), NULLPTR) != 0){
		const int errCode = errno;
		LOG_POSEIDON_WARNING("Error deleting from epoll: errno = ", errCode);
	}
	m_sessions->erase<IDX_ADDR>(it);
}
void Epoll::snapshot(std::vector<boost::shared_ptr<TcpSessionBase> > &sessions) const {
	const Mutex::UniqueLock lock(m_mutex);
	sessions.reserve(m_sessions->size());
	for(AUTO(it, m_sessions->begin()); it != m_sessions->end(); ++it){
		sessions.push_back(it->session);
	}
}
void Epoll::clear(){
	const Mutex::UniqueLock lock(m_mutex);
	AUTO(it, m_sessions->begin());
	while(it != m_sessions->end()){
		if(::epoll_ctl(m_epoll.get(), EPOLL_CTL_DEL, it->session->getFd(), NULLPTR) != 0){
			const int errCode = errno;
			LOG_POSEIDON_WARNING("Error deleting from epoll: errno = ", errCode);
		}
		it = m_sessions->erase(it);
	}
}

std::size_t Epoll::wait(unsigned timeout) NOEXCEPT {
	::epoll_event events[MAX_PUMP_COUNT];
	const int count = ::epoll_wait(m_epoll.get(), events, MAX_PUMP_COUNT, (int)timeout);
	if(count < 0){
		const int errCode = errno;
		if(errCode != EINTR){
			LOG_POSEIDON_ERROR("::epoll_wait() failed: errno = ", errCode);
		}
		return 0;
	}

	const AUTO(now, getFastMonoClock());
	for(unsigned i = 0; i < (unsigned)count; ++i){
		const AUTO_REF(event, events[i]);

		boost::shared_ptr<TcpSessionBase> session;
		SessionMap::delegated_container::nth_index<IDX_ADDR>::type::iterator it;
		{
			const Mutex::UniqueLock lock(m_mutex);
			it = m_sessions->find<IDX_ADDR>(static_cast<TcpSessionBase *>(event.data.ptr));
			if(it == m_sessions->end<IDX_ADDR>()){
				LOG_POSEIDON_WARNING("Session is not in epoll?");
				continue;
			}
			session = it->session;
		}

		if(event.events & EPOLLERR){
			int errCode;
			::socklen_t errLen = sizeof(errCode);
			if(::getsockopt(session->getFd(), SOL_SOCKET, SO_ERROR, &errCode, &errLen) != 0){
				errCode = errno;
			}
			const AUTO(desc, getErrorDesc(errCode));
			LOG_POSEIDON_WARNING("Socket error: ", desc);
			session->onClose();
			removeSession(session);
			continue;
		}

		if(event.events & EPOLLRDHUP){
			if(!it->readHup){
				session->shutdownRead();
				session->onReadHup();

				it->readHup = true;
			}
		}
		if(event.events & EPOLLHUP){
			if(!it->writeHup){
				session->shutdownWrite();
				session->onWriteHup();

				it->writeHup = true;
			}
		}
		if(it->readHup && it->writeHup){
			try {
				LOG_POSEIDON_INFO("Socket closed, remote is ", session->getRemoteInfo());
			} catch(...){
				LOG_POSEIDON_INFO("Socket closed, remote is not connected.");
			}
			session->onClose();
			removeSession(session);
			continue;
		}

		if(event.events & EPOLLIN){
			const Mutex::UniqueLock lock(m_mutex);
			m_sessions->setKey<IDX_ADDR, IDX_READ>(it, now);
		}
		if(event.events & EPOLLOUT){
			Mutex::UniqueLock sessionLock;
			if(!session->isSendBufferEmpty(sessionLock)){
				const Mutex::UniqueLock lock(m_mutex);
				m_sessions->setKey<IDX_ADDR, IDX_WRITE>(it, now);
			}
		}
	}
	return (unsigned)count;
}
std::size_t Epoll::pumpReadable(){
	// 有序的关系型容器在插入元素时迭代器不失效。这一点非常重要。
	SessionMap::delegated_container::nth_index<IDX_READ>::type::iterator iterators[MAX_PUMP_COUNT];
	std::size_t count = 0;
	{
		const Mutex::UniqueLock lock(m_mutex);
		for(AUTO(it, m_sessions->upperBound<IDX_READ>(0)); it != m_sessions->end<IDX_READ>(); ++it){
			iterators[count] = it;
			if(++count >= MAX_PUMP_COUNT){
				break;
			}
		}
	}
	for(std::size_t i = 0; i < count; ++i){
		const AUTO_REF(it, iterators[i]);
		const AUTO(session, it->session);

		try {
			unsigned char temp[1024];
			const AUTO(result, session->syncReadAndProcess(temp, sizeof(temp)));
			if(result.bytesTransferred < 0){
				if(result.errCode == EINTR){
					continue;
				}
				if(result.errCode == EAGAIN){
					const Mutex::UniqueLock lock(m_mutex);
					m_sessions->setKey<IDX_READ, IDX_READ>(it, 0);
					continue;
				}
				DEBUG_THROW(SystemException);
			} else if(result.bytesTransferred == 0){
				const Mutex::UniqueLock lock(m_mutex);
				m_sessions->setKey<IDX_READ, IDX_READ>(it, 0);
				continue;
			}
		} catch(std::exception &e){
			LOG_POSEIDON_WARNING("std::exception thrown while dispatching data: what = ", e.what());
			session->forceShutdown();
		} catch(...){
			LOG_POSEIDON_WARNING("Unknown exception thrown while dispatching data.");
			session->forceShutdown();
		}
	}
	return count;
}
std::size_t Epoll::pumpWriteable(){
	// 有序的关系型容器在插入元素时迭代器不失效。这一点非常重要。
	SessionMap::delegated_container::nth_index<IDX_WRITE>::type::iterator iterators[MAX_PUMP_COUNT];
	std::size_t count = 0;
	{
		const Mutex::UniqueLock lock(m_mutex);
		for(AUTO(it, m_sessions->upperBound<IDX_WRITE>(0)); it != m_sessions->end<IDX_WRITE>(); ++it){
			iterators[count] = it;
			if(++count >= MAX_PUMP_COUNT){
				break;
			}
		}
	}
	for(std::size_t i = 0; i < count; ++i){
		const AUTO_REF(it, iterators[i]);
		const AUTO(session, it->session);

		try {
			unsigned char temp[1024];
			const AUTO(result, session->syncWrite(temp, sizeof(temp)));
			if(result.bytesTransferred < 0){
				if(result.errCode == EINTR){
					continue;
				}
				if(result.errCode == EAGAIN){
					const Mutex::UniqueLock lock(m_mutex);
					m_sessions->setKey<IDX_WRITE, IDX_WRITE>(it, 0);
					continue;
				}
				DEBUG_THROW(SystemException);
			} else if(result.bytesTransferred == 0){
				Mutex::UniqueLock sessionLock;
				if(session->isSendBufferEmpty(sessionLock)){
					const Mutex::UniqueLock lock(m_mutex);
					m_sessions->setKey<IDX_WRITE, IDX_WRITE>(it, 0);
				}
				continue;
			}
		} catch(std::exception &e){
			LOG_POSEIDON_WARNING("std::exception thrown while writing socket: what = ", e.what());
			session->forceShutdown();
		} catch(...){
			LOG_POSEIDON_WARNING("Unknown exception thrown while writing socket.");
			session->forceShutdown();
		}
	}
	return count;
}

}
