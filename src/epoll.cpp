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
		TcpSessionBase *addr;
		// 时间戳，零表示无数据可读/写。
		boost::uint64_t lastRead;
		boost::uint64_t lastWritten;

		boost::shared_ptr<TcpSessionBase> session;
		boost::shared_ptr<const boost::weak_ptr<Epoll> > epoll;

		SessionMapElement(boost::shared_ptr<TcpSessionBase> session_, boost::uint64_t lastRead_, boost::uint64_t lastWritten_,
			boost::weak_ptr<Epoll> epoll_)
			: addr(session_.get()), lastRead(lastRead_), lastWritten(lastWritten_)
			, session(STD_MOVE(session_)), epoll(boost::make_shared<boost::weak_ptr<Epoll> >(STD_MOVE(epoll_)))
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

void Epoll::notifyWriteable(TcpSessionBase *session){
	const AUTO(now, getFastMonoClock());
	const boost::mutex::scoped_lock lock(m_mutex);
	const AUTO(it, m_sessions->find<IDX_ADDR>(session));
	if(it == m_sessions->end<IDX_ADDR>()){
		LOG_POSEIDON_WARNING("Session is not in epoll?");
		return;
	}
	m_sessions->setKey<IDX_ADDR, IDX_WRITE>(it, now);
}
void Epoll::notifyUnlinked(TcpSessionBase *session){
	const boost::mutex::scoped_lock lock(m_mutex);
	const AUTO(it, m_sessions->find<IDX_ADDR>(session));
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

void Epoll::addSession(const boost::shared_ptr<TcpSessionBase> &session){
	const boost::mutex::scoped_lock lock(m_mutex);
	const AUTO(result, m_sessions->insert(SessionMapElement(session, 0, 0, shared_from_this())));
	if(!result.second){
		LOG_POSEIDON_WARNING("Session is already in epoll.");
		return;
	}
	::epoll_event event;
	event.events = static_cast< ::uint32_t>(EPOLLIN | EPOLLOUT | EPOLLET);
	event.data.ptr = session.get();
	if(::epoll_ctl(m_epoll.get(), EPOLL_CTL_ADD, session->getFd(), &event) != 0){
		const int errCode = errno;
		m_sessions->erase(result.first); // !!
		DEBUG_THROW(SystemException, errCode);
	}
	session->setEpoll(result.first->epoll);
}
void Epoll::removeSession(const boost::shared_ptr<TcpSessionBase> &session){
	const boost::mutex::scoped_lock lock(m_mutex);
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
	const boost::mutex::scoped_lock lock(m_mutex);
	sessions.reserve(m_sessions->size());
	for(AUTO(it, m_sessions->begin()); it != m_sessions->end(); ++it){
		sessions.push_back(it->session);
	}
}
void Epoll::clear(){
	const boost::mutex::scoped_lock lock(m_mutex);
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
	const int count = ::epoll_wait(m_epoll.get(), events, (int)COUNT_OF(events), (int)timeout);
	if(count < 0){
		const int errCode = errno;
		if(errCode != EINTR){
			LOG_POSEIDON_ERROR("::epoll_wait() failed: errno = ", errCode);
		}
		return 0;
	}

	const AUTO(now, getFastMonoClock());
	for(unsigned i = 0; i < (unsigned)count; ++i){
		const ::epoll_event &event = events[i];

		boost::shared_ptr<TcpSessionBase> session;
		SessionMap::delegated_container::nth_index<IDX_ADDR>::type::iterator it;
		{
			const boost::mutex::scoped_lock lock(m_mutex);
			it = m_sessions->find<IDX_ADDR>(static_cast<TcpSessionBase *>(event.data.ptr));
			if(it == m_sessions->end<IDX_ADDR>()){
				LOG_POSEIDON_WARNING("Session is not in epoll?");
				continue;
			}
			session = it->session;
		}

		if(event.events & EPOLLHUP){
			try {
				LOG_POSEIDON_INFO("Socket hung up, remote is ", session->getRemoteInfo());
			} catch(...){
				LOG_POSEIDON_INFO("Socket hung up, remote is not connected.");
			}
			session->onClose();
			removeSession(session);
			continue;
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

		{
			boost::mutex::scoped_lock lock(m_mutex, boost::defer_lock);
			if(event.events & EPOLLIN){
//				if(!lock.owns_lock()){
					lock.lock();
//				}
				m_sessions->setKey<IDX_ADDR, IDX_READ>(it, now);
			}
			if(event.events & EPOLLOUT){
				if(!lock.owns_lock()){
					lock.lock();
				}
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
		const boost::mutex::scoped_lock lock(m_mutex);
		for(AUTO(it, m_sessions->upperBound<IDX_READ>(0)); it != m_sessions->end<IDX_READ>(); ++it){
			iterators[count] = it;
			if(++count >= MAX_PUMP_COUNT){
				break;
			}
		}
	}
	for(std::size_t i = 0; i < count; ++i){
		const AUTO_REF(it, iterators[i]);
		const AUTO_REF(session, it->session);

		try {
			unsigned char temp[1024];
			long bytesRead = session->syncReadAndProcess(temp, sizeof(temp));
			if(bytesRead < 0){
				if(errno == EINTR){
					continue;
				}
				if(errno == EAGAIN){
					const boost::mutex::scoped_lock lock(m_mutex);
					m_sessions->setKey<IDX_READ, IDX_READ>(it, 0);
					continue;
				}
				DEBUG_THROW(SystemException);
			} else if(bytesRead == 0){
				if(session->shutdown()){
					LOG_POSEIDON_INFO("Connection closed: remote = ", session->getRemoteInfo());
					session->onReadHup();
				}
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
		const boost::mutex::scoped_lock lock(m_mutex);
		for(AUTO(it, m_sessions->upperBound<IDX_WRITE>(0)); it != m_sessions->end<IDX_WRITE>(); ++it){
			iterators[count] = it;
			if(++count >= MAX_PUMP_COUNT){
				break;
			}
		}
	}
	for(std::size_t i = 0; i < count; ++i){
		const AUTO_REF(it, iterators[i]);
		const AUTO_REF(session, it->session);

		try {
			unsigned char temp[1024];
			long bytesWritten;
			bool readHup;
			{
				boost::mutex::scoped_lock sessionLock;
				bytesWritten = session->syncWrite(sessionLock, temp, sizeof(temp));
				readHup = session->hasBeenShutdown();
				if(bytesWritten == 0){
					if(!readHup){
						const boost::mutex::scoped_lock lock(m_mutex);
						m_sessions->setKey<IDX_WRITE, IDX_WRITE>(it, 0);
					}
				}
			}
			if(bytesWritten < 0){
				if(errno == EINTR){
					continue;
				}
				if(errno == EAGAIN){
					const boost::mutex::scoped_lock lock(m_mutex);
					m_sessions->setKey<IDX_WRITE, IDX_WRITE>(it, 0);
					continue;
				}
				DEBUG_THROW(SystemException);
			} else if(bytesWritten == 0){
				if(readHup){
					const TcpSessionBase::DelayedShutdownGuard guard(session);
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
