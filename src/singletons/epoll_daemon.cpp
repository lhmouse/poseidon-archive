// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2017, LH_Mouse. All wrongs reserved.

#include "../precompiled.hpp"
#include "epoll_daemon.hpp"
#include <sys/types.h>
#include <sys/socket.h>
#include <sys/epoll.h>
#include <errno.h>
#include "job_dispatcher.hpp"
#include "../thread.hpp"
#include "../log.hpp"
#include "../atomic.hpp"
#include "../time.hpp"
#include "../socket_base.hpp"
#include "../profiler.hpp"
#include "../recursive_mutex.hpp"
#include "../raii.hpp"
#include "../multi_index_map.hpp"
#include "../checked_arithmetic.hpp"
#include "../system_exception.hpp"
#include "../errno.hpp"

namespace Poseidon {

namespace {
	volatile bool g_running = false;
	Thread g_thread;

	struct SocketElement {
		boost::shared_ptr<SocketBase> socket;

		int fd;
		boost::uint64_t read_time;
		boost::uint64_t write_time;

		mutable bool readable;
		mutable bool writeable;

		explicit SocketElement(boost::shared_ptr<SocketBase> socket_)
			: socket(STD_MOVE_IDN(socket_))
			, fd(socket->get_fd()), read_time((boost::uint64_t)-1), write_time((boost::uint64_t)-1)
			, readable(false), writeable(false)
		{
		}
	};
	MULTI_INDEX_MAP(SocketMap, SocketElement,
		UNIQUE_MEMBER_INDEX(fd)
		MULTI_MEMBER_INDEX(read_time)
		MULTI_MEMBER_INDEX(write_time)
	)

	RecursiveMutex g_mutex;
	UniqueFile g_epoll;
	SocketMap g_socket_map;

	bool pump_readable_sockets() NOEXCEPT {
		PROFILE_ME;

		std::vector<VALUE_TYPE(g_socket_map.begin<1>())> iterators;
		const AUTO(now, get_fast_mono_clock());
		{
			const RecursiveMutex::UniqueLock lock(g_mutex);
			const AUTO(range, std::make_pair(g_socket_map.begin<1>(), g_socket_map.upper_bound<1>(now)));
			try {
				iterators.reserve(static_cast<std::size_t>(std::distance(range.first, range.second)));
				for(AUTO(it, range.first); it != range.second; ++it){
					iterators.push_back(it);
				}
			} catch(std::exception &e){
				LOG_POSEIDON_ERROR("std::exception thrown: what = ", e.what());
				return false;
			}
		}
		if(iterators.empty()){
			return false;
		}
		for(AUTO(iit, iterators.begin()); iit != iterators.end(); ++iit){
			const AUTO(it, *iit);
			if(it->socket->is_throttled()){
				LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_DEBUG,
					"Session is throttled: typeid = ", typeid(*(it->socket)).name());
				const RecursiveMutex::UniqueLock lock(g_mutex);
				g_socket_map.set_key<1, 1>(it, now + 5000);
				continue;
			}
			try {
				const int err_code = it->socket->poll_read_and_process(it->readable);
				if((err_code != 0) && (err_code != EINTR)){
					if(err_code == EWOULDBLOCK){
						const RecursiveMutex::UniqueLock lock(g_mutex);
						g_socket_map.set_key<1, 1>(it, (boost::uint64_t)-1);
						continue;
					}
					DEBUG_THROW(SystemException, err_code);
				}
			} catch(std::exception &e){
				LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_INFO,
					"std::exception thrown: what = ", e.what(), ", typeid = ", typeid(*(it->socket)).name());
				it->socket->force_shutdown();
				const RecursiveMutex::UniqueLock lock(g_mutex);
				g_socket_map.erase<1>(it);
				continue;
			} catch(...){
				LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_INFO,
					"Unknown exception thrown: typeid = ", typeid(*(it->socket)).name());
				it->socket->force_shutdown();
				const RecursiveMutex::UniqueLock lock(g_mutex);
				g_socket_map.erase<1>(it);
				continue;
			}
		}
		return true;
	}

	bool pump_writeable_sockets() NOEXCEPT {
		PROFILE_ME;

		std::vector<VALUE_TYPE(g_socket_map.begin<2>())> iterators;
		const AUTO(now, get_fast_mono_clock());
		{
			const RecursiveMutex::UniqueLock lock(g_mutex);
			const AUTO(range, std::make_pair(g_socket_map.begin<2>(), g_socket_map.upper_bound<2>(now)));
			try {
				iterators.reserve(static_cast<std::size_t>(std::distance(range.first, range.second)));
				for(AUTO(it, range.first); it != range.second; ++it){
					iterators.push_back(it);
				}
			} catch(std::exception &e){
				LOG_POSEIDON_ERROR("std::exception thrown: what = ", e.what());
				return false;
			}
		}
		if(iterators.empty()){
			return false;
		}
		for(AUTO(iit, iterators.begin()); iit != iterators.end(); ++iit){
			const AUTO(it, *iit);
			try {
				Mutex::UniqueLock write_lock;
				const int err_code = it->socket->poll_write(write_lock, it->writeable);
				if((err_code != 0) && (err_code != EINTR)){
					if(err_code == EWOULDBLOCK){
						const RecursiveMutex::UniqueLock lock(g_mutex);
						g_socket_map.set_key<2, 2>(it, (boost::uint64_t)-1);
						continue;
					}
					DEBUG_THROW(SystemException, err_code);
				}
			} catch(std::exception &e){
				LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_INFO,
					"std::exception thrown: what = ", e.what(), ", typeid = ", typeid(*(it->socket)).name());
				it->socket->force_shutdown();
				const RecursiveMutex::UniqueLock lock(g_mutex);
				g_socket_map.erase<2>(it);
				continue;
			} catch(...){
				LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_INFO,
					"Unknown exception thrown: typeid = ", typeid(*(it->socket)).name());
				it->socket->force_shutdown();
				const RecursiveMutex::UniqueLock lock(g_mutex);
				g_socket_map.erase<2>(it);
				continue;
			}
		}
		return true;
	}

	bool wait_for_sockets(unsigned timeout) NOEXCEPT {
		PROFILE_ME;

		::epoll_event events[256];
		const int result = ::epoll_wait(g_epoll.get(), events, COUNT_OF(events), (int)timeout);
		if(result < 0){
			const int err_code = errno;
			if(err_code != EINTR){
				LOG_POSEIDON_ERROR("::epoll_wait() failed! errno was ", err_code);
			}
			return false;
		}
		if(result == 0){
			return false;
		}
		const AUTO(now, Poseidon::get_fast_mono_clock());
		const RecursiveMutex::UniqueLock lock(g_mutex);
		for(unsigned i = 0; i < (unsigned)result; ++i){
			const AUTO(it, g_socket_map.find<0>(events[i].data.fd));
			if(it == g_socket_map.end()){
				LOG_POSEIDON_DEBUG("Socket reported by epoll is not registered: fd = ", events[i].data.fd);
				continue;
			}
			if(events[i].events & (EPOLLHUP | EPOLLERR)){
				int err_code;
				if(it->socket->was_timed_out()){
					err_code = ETIMEDOUT;
				} else if(events[i].events & EPOLLERR){
					::socklen_t err_len = sizeof(err_code);
					if(::getsockopt(it->socket->get_fd(), SOL_SOCKET, SO_ERROR, &err_code, &err_len) != 0){
						err_code = errno;
						LOG_POSEIDON_WARNING("::getsockopt() failed, errno was ", err_code, ": fd = ", it->socket->get_fd());
					}
				} else {
					err_code = 0;
				}
				LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_DEBUG,
					"Socket closed: err_code = ", err_code, ", desc = ", get_error_desc(err_code), ", typeid = ", typeid(*(it->socket)).name());
				it->socket->on_close(err_code);
				g_socket_map.erase<0>(it);
				continue;
			}
			if(events[i].events & EPOLLIN){
				it->readable = true;
				g_socket_map.set_key<0, 1>(it, now);
			}
			if(events[i].events & EPOLLOUT){
				it->writeable = true;
				g_socket_map.set_key<0, 2>(it, now);
			}
		}
		return true;
	}

	void thread_proc(){
		PROFILE_ME;
		LOG_POSEIDON_INFO("Epoll daemon started.");

		unsigned timeout = 0;
		for(;;){
			bool busy;
			do {
				busy = wait_for_sockets(0);
				busy += JobDispatcher::is_running() && pump_readable_sockets();
				busy += pump_writeable_sockets();
				timeout = std::min(timeout * 2u + 1u, !busy * 100u);
			} while(busy);

			if(!atomic_load(g_running, ATOMIC_CONSUME)){
				break;
			}
			wait_for_sockets(timeout);
		}

		LOG_POSEIDON_INFO("Epoll daemon stopped.");
	}
}

void EpollDaemon::start(){
	if(atomic_exchange(g_running, true, ATOMIC_ACQ_REL) != false){
		LOG_POSEIDON_FATAL("Only one daemon is allowed at the same time.");
		std::abort();
	}
	LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_INFO, "Starting epoll daemon...");

	if(!g_epoll.reset(::epoll_create(4096))){
		const int err_code = errno;
		LOG_POSEIDON_FATAL("Failed to create epoll! errno was ", err_code);
		std::abort();
	}
	Thread(&thread_proc, "   N").swap(g_thread);
}
void EpollDaemon::stop(){
	if(atomic_exchange(g_running, false, ATOMIC_ACQ_REL) == false){
		return;
	}
	LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_INFO, "Stopping epoll daemon...");

	if(g_thread.joinable()){
		g_thread.join();
	}
	g_socket_map.clear();
	g_epoll.reset();
}

void EpollDaemon::make_snapshot(std::vector<EpollDaemon::SnapshotElement> &snapshot){
	PROFILE_ME;

	const AUTO(now, get_fast_mono_clock());
	const RecursiveMutex::UniqueLock lock(g_mutex);
	snapshot.reserve(snapshot.size() + g_socket_map.size());
	for(AUTO(it, g_socket_map.begin()); it != g_socket_map.end(); ++it){
		SnapshotElement elem;
		elem.remote = it->socket->get_remote_info();
		elem.local = it->socket->get_local_info();
		elem.ms_online = saturated_sub(now, it->socket->get_creation_time());
		snapshot.push_back(STD_MOVE(elem));
	}
}
void EpollDaemon::add_socket(const boost::shared_ptr<SocketBase> &socket){
	PROFILE_ME;

	const RecursiveMutex::UniqueLock lock(g_mutex);
	const AUTO(result, g_socket_map.insert(SocketElement(socket)));
	if(!result.second){
		LOG_POSEIDON_ERROR("Socket is already in epoll: socket = ", socket,
			", typeid = ", typeid(*socket).name(), ", fd = ", socket->get_fd());
		DEBUG_THROW(Exception, sslit("Socket is already in epoll"));
	}
	try {
		::epoll_event event = { };
		event.events = static_cast< ::uint32_t>(EPOLLIN | EPOLLOUT | EPOLLET);
		event.data.fd = socket->get_fd();
		if(::epoll_ctl(g_epoll.get(), EPOLL_CTL_ADD, socket->get_fd(), &event) != 0){
			const int err_code = errno;
			LOG_POSEIDON_ERROR("::epoll_ctl() failed, errno was ", err_code, ": socket = ", socket,
				", typeid = ", typeid(*socket).name(), ", fd = ", socket->get_fd());
			DEBUG_THROW(SystemException, err_code);
		}
	} catch(...){
		g_socket_map.erase(result.first);
		throw;
	}
}
bool EpollDaemon::mark_socket_writeable(int fd) NOEXCEPT {
	PROFILE_ME;

	const RecursiveMutex::UniqueLock lock(g_mutex);
	const AUTO(it, g_socket_map.find<0>(fd));
	if(it == g_socket_map.end()){
		LOG_POSEIDON_DEBUG("Socket not found in epoll: fd = ", fd);
		return false;
	}
	const AUTO(now, get_fast_mono_clock());
	g_socket_map.set_key<0, 2>(it, now);
	return true;
}

}
