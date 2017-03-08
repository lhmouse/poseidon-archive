// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2017, LH_Mouse. All wrongs reserved.

#include "precompiled.hpp"
#include "epoll.hpp"
#include "tcp_session_base.hpp"
#include <typeinfo>
#include <sys/epoll.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <errno.h>
#include "multi_index_map.hpp"
#include "system_exception.hpp"
#include "log.hpp"
#include "time.hpp"
#include "errno.hpp"
#include "atomic.hpp"
#include "profiler.hpp"

namespace Poseidon {

namespace {
	enum {
		MAX_EPOLL_PUMP_COUNT    = 256,
		IO_BUFFER_SIZE          = 4096,
		MAX_SEND_BUFFER_SIZE    = 65536,

		THROTTLED_RETRY_DELAY   = 5000,
	};

	CONSTEXPR const AUTO(TIME_POINT_MAX, static_cast<boost::uint64_t>(-1));

	struct SessionMapElement {
		boost::shared_ptr<TcpSessionBase> session;

		TcpSessionBase *addr;
		boost::uint64_t next_read;
		boost::uint64_t next_write;

		explicit SessionMapElement(boost::shared_ptr<TcpSessionBase> session_)
			: session(STD_MOVE(session_))
			, addr(session.get()), next_read(TIME_POINT_MAX), next_write(TIME_POINT_MAX)
		{
		}
	};
}

MULTI_INDEX_MAP(Epoll::SessionMap, SessionMapElement,
	UNIQUE_MEMBER_INDEX(addr)
	MULTI_MEMBER_INDEX(next_read)
	MULTI_MEMBER_INDEX(next_write)
)

Epoll::Epoll()
	: m_mutex()
{
	if(!m_epoll.reset(::epoll_create(4096))){
		DEBUG_THROW(SystemException);
	}
	m_sessions.reset(new SessionMap);
}
Epoll::~Epoll(){
}

void Epoll::notify_writeable(TcpSessionBase *session) NOEXCEPT {
	PROFILE_ME;

	const AUTO(now, get_fast_mono_clock());
	const RecursiveMutex::UniqueLock lock(m_mutex);
	const AUTO(it, m_sessions->find<0>(session));
	if(it == m_sessions->end<0>()){
		LOG_POSEIDON_DEBUG("Session is no longer in epoll.");
		return;
	}
	m_sessions->set_key<0, 2>(it, now);
}
void Epoll::notify_unlinked(TcpSessionBase *session) NOEXCEPT {
	PROFILE_ME;

	const RecursiveMutex::UniqueLock lock(m_mutex);
	const AUTO(it, m_sessions->find<0>(session));
	if(it == m_sessions->end<0>()){
		LOG_POSEIDON_WARNING("Session is not in epoll.");
		return;
	}
	if(::epoll_ctl(m_epoll.get(), EPOLL_CTL_DEL, session->get_fd(), (::epoll_event *)-1) != 0){
		const int err_code = errno;
		LOG_POSEIDON_WARNING("Error deleting from epoll: errno = ", err_code);
	}
	m_sessions->erase<0>(it);
}

void Epoll::add_session(const boost::shared_ptr<TcpSessionBase> &session){
	PROFILE_ME;

	boost::weak_ptr<Epoll> weak_this(shared_from_this());

	const RecursiveMutex::UniqueLock lock(m_mutex);
	const AUTO(result, m_sessions->insert(SessionMapElement(session)));
	if(!result.second){
		LOG_POSEIDON_WARNING("Session is already in epoll.");
		return;
	}
	::epoll_event event;
	event.events = static_cast< ::uint32_t>(EPOLLIN | EPOLLOUT | EPOLLET);
	event.data.ptr = session.get();
	if(::epoll_ctl(m_epoll.get(), EPOLL_CTL_ADD, session->get_fd(), &event) != 0){
		const int err_code = errno;
		m_sessions->erase(result.first); // !!
		DEBUG_THROW(SystemException, err_code);
	}
	session->set_epoll(STD_MOVE(weak_this));
}
void Epoll::remove_session(const boost::shared_ptr<TcpSessionBase> &session){
	PROFILE_ME;

	const RecursiveMutex::UniqueLock lock(m_mutex);
	const AUTO(it, m_sessions->find<0>(session.get()));
	if(it == m_sessions->end<0>()){
		LOG_POSEIDON_WARNING("Session is not in epoll.");
		return;
	}
	if(::epoll_ctl(m_epoll.get(), EPOLL_CTL_DEL, session->get_fd(), NULLPTR) != 0){
		const int err_code = errno;
		LOG_POSEIDON_WARNING("Error deleting from epoll: errno = ", err_code);
	}
	m_sessions->erase<0>(it);
}
void Epoll::snapshot(std::vector<boost::shared_ptr<TcpSessionBase> > &sessions) const {
	PROFILE_ME;

	const RecursiveMutex::UniqueLock lock(m_mutex);
	sessions.reserve(m_sessions->size());
	for(AUTO(it, m_sessions->begin()); it != m_sessions->end(); ++it){
		sessions.push_back(it->session);
	}
}
void Epoll::clear(){
	PROFILE_ME;

	const RecursiveMutex::UniqueLock lock(m_mutex);
	AUTO(it, m_sessions->begin());
	while(it != m_sessions->end()){
		if(::epoll_ctl(m_epoll.get(), EPOLL_CTL_DEL, it->session->get_fd(), NULLPTR) != 0){
			const int err_code = errno;
			LOG_POSEIDON_WARNING("Error deleting from epoll: errno = ", err_code);
		}
		it = m_sessions->erase(it);
	}
}

std::size_t Epoll::wait(unsigned timeout) NOEXCEPT {
	PROFILE_ME;

	::epoll_event events[MAX_EPOLL_PUMP_COUNT];
	const int count = ::epoll_wait(m_epoll.get(), events, COUNT_OF(events), (int)timeout);
	if(count < 0){
		const int err_code = errno;
		if(err_code != EINTR){
			LOG_POSEIDON_ERROR("::epoll_wait() failed: errno = ", err_code);
		}
		return 0;
	}

	const AUTO(now, get_fast_mono_clock());
	for(unsigned i = 0; i < (unsigned)count; ++i){
		const AUTO_REF(event, events[i]);

		boost::shared_ptr<TcpSessionBase> session;
		SessionMap::base_container::nth_index<0>::type::iterator it;
		{
			const RecursiveMutex::UniqueLock lock(m_mutex);
			it = m_sessions->find<0>(static_cast<TcpSessionBase *>(event.data.ptr));
			if(it == m_sessions->end<0>()){
				LOG_POSEIDON_WARNING("Session is not in epoll?");
				continue;
			}
			session = it->session;
		}

		if(event.events & EPOLLERR){
			int err_code;
			if(atomic_load(session->m_timed_out, ATOMIC_CONSUME)){
				err_code = ETIMEDOUT;
			} else {
				::socklen_t err_len = sizeof(err_code);
				if(::getsockopt(session->get_fd(), SOL_SOCKET, SO_ERROR, &err_code, &err_len) != 0){
					err_code = errno;
				}
			}
			const AUTO(desc, get_error_desc(err_code));
			LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_DEBUG,
				"Socket error: err_code = ", err_code, ", desc = ", desc, ", typeid = ", typeid(*session).name(),
				", remote = ", session->get_remote_info_nothrow());
			session->shutdown_read();
			session->shutdown_write();
			session->on_close(err_code);
			goto _erase_session;
		}
		if(event.events & EPOLLHUP){
			LOG_POSEIDON_INFO("Socket closed: remote = ", session->get_remote_info_nothrow());
			LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_DEBUG, "Socket closed gracefully: typeid = ", typeid(*session).name());
			session->shutdown_read();
			session->shutdown_write();
			session->on_close(0);
			goto _erase_session;
		}

		if(event.events & EPOLLIN){
			if(!session->is_read_hup_notified()){
				const RecursiveMutex::UniqueLock lock(m_mutex);
				m_sessions->set_key<0, 1>(it, now);
			}
		}
		if(event.events & EPOLLOUT){
			session->set_connected();

			if((session->get_send_buffer_size() != 0) || !session->is_connected_notified()){
				const RecursiveMutex::UniqueLock lock(m_mutex);
				m_sessions->set_key<0, 2>(it, now);
			}
		}
		continue;

	_erase_session:
		const RecursiveMutex::UniqueLock lock(m_mutex);
		if(::epoll_ctl(m_epoll.get(), EPOLL_CTL_DEL, session->get_fd(), NULLPTR) != 0){
			const int err_code = errno;
			LOG_POSEIDON_WARNING("Error deleting from epoll: errno = ", err_code);
		}
		m_sessions->erase<0>(it);
	}
	return (unsigned)count;
}
std::size_t Epoll::pump_readable(){
	PROFILE_ME;

	const AUTO(now, get_fast_mono_clock());

	// 有序的关系型容器在插入元素时迭代器不失效。这一点非常重要。
	std::vector<VALUE_TYPE(m_sessions->begin<1>())> iterators;
	iterators.reserve(MAX_EPOLL_PUMP_COUNT);
	{
		const RecursiveMutex::UniqueLock lock(m_mutex);
		const AUTO(range, std::make_pair(m_sessions->begin<1>(), m_sessions->upper_bound<1>(now)));
		for(AUTO(it, range.first); it != range.second; ++it){
			iterators.push_back(it);
		}
	}
	for(AUTO(iit, iterators.begin()); iit != iterators.end(); ++iit){
		const AUTO(it, *iit);
		const AUTO(session, it->session);

		try {
			if(session->is_throttled()){
				LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_DEBUG,
					"Session is throttled: typeid = ", typeid(*session).name());

				const RecursiveMutex::UniqueLock lock(m_mutex);
				m_sessions->set_key<1, 1>(it, now + THROTTLED_RETRY_DELAY);
				continue;
			}

			if(session->get_send_buffer_size() > MAX_SEND_BUFFER_SIZE){
				LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_DEBUG,
					"Max send buffer size exceeded: typeid = ", typeid(*session).name());

				const RecursiveMutex::UniqueLock lock(m_mutex);
				m_sessions->set_key<1, 1>(it, now + THROTTLED_RETRY_DELAY);
				continue;
			}

			unsigned char temp[IO_BUFFER_SIZE];
			const AUTO(result, session->sync_read_and_process(temp, sizeof(temp)));
			if(result.bytes_transferred < 0){
				if(result.err_code == EINTR){
					continue;
				}
				if(result.err_code == EAGAIN){
					const RecursiveMutex::UniqueLock lock(m_mutex);
					m_sessions->set_key<1, 1>(it, TIME_POINT_MAX);
					continue;
				}
				DEBUG_THROW(SystemException, result.err_code);
			} else if(result.bytes_transferred == 0){
				session->notify_read_hup();

				const RecursiveMutex::UniqueLock lock(m_mutex);
				m_sessions->set_key<1, 1>(it, TIME_POINT_MAX);
				continue;
			}
		} catch(std::exception &e){
			LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_INFO,
				"std::exception thrown while reading socket: what = ", e.what(), ", typeid = ", typeid(*session).name());
			session->force_shutdown();
		} catch(...){
			LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_INFO,
				"Unknown exception thrown while reading socket: typeid = ", typeid(*session).name());
			session->force_shutdown();
		}
	}
	return iterators.size();
}
std::size_t Epoll::pump_writeable(){
	PROFILE_ME;

	const AUTO(now, get_fast_mono_clock());

	// 有序的关系型容器在插入元素时迭代器不失效。这一点非常重要。
	std::vector<VALUE_TYPE(m_sessions->begin<2>())> iterators;
	iterators.reserve(MAX_EPOLL_PUMP_COUNT);
	{
		const RecursiveMutex::UniqueLock lock(m_mutex);
		const AUTO(range, std::make_pair(m_sessions->begin<2>(), m_sessions->upper_bound<2>(now)));
		for(AUTO(it, range.first); it != range.second; ++it){
			iterators.push_back(it);
		}
	}
	for(AUTO(iit, iterators.begin()); iit != iterators.end(); ++iit){
		const AUTO(it, *iit);
		const AUTO(session, it->session);

		try {
			if(session->is_connected()){
				session->notify_connected();
			}

			unsigned char temp[IO_BUFFER_SIZE];
			const AUTO(result, session->sync_write(temp, sizeof(temp)));
			if(result.bytes_transferred < 0){
				if(result.err_code == EINTR){
					continue;
				}
				if(result.err_code == EAGAIN){
					const RecursiveMutex::UniqueLock lock(m_mutex);
					m_sessions->set_key<2, 2>(it, TIME_POINT_MAX);
					continue;
				}
				DEBUG_THROW(SystemException, result.err_code);
			} else if(result.bytes_transferred == 0){
				Mutex::UniqueLock session_lock;
				if(session->get_send_buffer_size(&session_lock) == 0){
					const RecursiveMutex::UniqueLock lock(m_mutex);
					m_sessions->set_key<2, 2>(it, TIME_POINT_MAX);
				}
				continue;
			}
		} catch(std::exception &e){
			LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_INFO,
				"std::exception thrown while writing socket: what = ", e.what(), ", typeid = ", typeid(*session).name());
			session->force_shutdown();
		} catch(...){
			LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_INFO,
				"Unknown exception thrown while writing socket: typeid = ", typeid(*session).name());
			session->force_shutdown();
		}
	}
	return iterators.size();
}

}
