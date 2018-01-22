// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2018, LH_Mouse. All wrongs reserved.

#include "../precompiled.hpp"
#include "epoll_daemon.hpp"
#include "main_config.hpp"
#include <sys/types.h>
#include <sys/socket.h>
#include <sys/epoll.h>
#include <unistd.h>
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
#include "../flags.hpp"

namespace Poseidon {

namespace {
	volatile bool g_running = false;
	Thread g_thread;

	class WeakableSocket {
	private:
		boost::shared_ptr<SocketBase> m_strong;
		boost::weak_ptr<SocketBase> m_weak;
		UniqueFile m_dup_fd;

	public:
		WeakableSocket(bool owning, const boost::shared_ptr<SocketBase> &socket){
			if(owning){
				m_strong = socket;
			} else {
				m_weak = socket;
				DEBUG_THROW_ASSERT(m_dup_fd.reset(::dup(socket->get_fd())));
			}
		}

	public:
		boost::shared_ptr<SocketBase> lock() const NOEXCEPT {
			return m_strong ? m_strong : m_weak.lock();
		}
	};

	struct SocketElement {
		boost::shared_ptr<const WeakableSocket> weakable;

		const SocketBase *ptr;
		boost::uint64_t read_time;
		boost::uint64_t write_time;
		int err_code;

		mutable bool readable;
		mutable bool writeable;

		SocketElement(bool owning, const boost::shared_ptr<SocketBase> &socket, boost::uint64_t now)
			: weakable(boost::make_shared<WeakableSocket>(owning, socket))
			, ptr(socket.get()), read_time(now), write_time(now), err_code(-1)
			, readable(false), writeable(false)
		{ }
	};
	MULTI_INDEX_MAP(SocketMap, SocketElement,
		UNIQUE_MEMBER_INDEX(ptr)
		MULTI_MEMBER_INDEX(read_time)
		MULTI_MEMBER_INDEX(write_time)
		MULTI_MEMBER_INDEX(err_code)
	);

	RecursiveMutex g_mutex;
	UniqueFile g_epoll;
	SocketMap g_socket_map;

	bool wait_for_sockets(unsigned timeout) NOEXCEPT {
		PROFILE_ME;

		boost::array< ::epoll_event, 256> events;
		const int result = ::epoll_wait(g_epoll.get(), events.data(), static_cast<int>(events.size()), static_cast<int>(std::min<unsigned>(timeout, INT_MAX)));
		if(result < 0){
			const int err_code = errno;
			if(err_code != EINTR){
				LOG_POSEIDON_ERROR("::epoll_wait() failed! errno was ", err_code, " (", get_error_desc(err_code), ")");
			}
			return false;
		}
		if(result == 0){
			return false;
		}
		const AUTO(now, get_fast_mono_clock());
		const RecursiveMutex::UniqueLock lock(g_mutex);
		for(unsigned i = 0; i < static_cast<unsigned>(result); ++i){
			const AUTO(ptr, static_cast<SocketBase *>(events[i].data.ptr));
			const AUTO(it, g_socket_map.find<0>(ptr));
			if(it == g_socket_map.end()){
				LOG_POSEIDON_TRACE("Socket reported by epoll is not registered: ptr = ", static_cast<void *>(ptr));
				continue;
			}
			const AUTO(socket, it->weakable->lock());
			if(!socket){
				g_socket_map.erase<0>(it);
				continue;
			}
			if(has_any_flags_of(events[i].events, EPOLLIN) && !socket->has_been_shutdown_read()){
				it->readable += has_none_flags_of(events[i].events, EPOLLERR);
				g_socket_map.set_key<0, 1>(it, now);
			}
			if(has_any_flags_of(events[i].events, EPOLLOUT) && !socket->has_been_shutdown_write()){
				it->writeable += has_none_flags_of(events[i].events, EPOLLERR);
				g_socket_map.set_key<0, 2>(it, now);
			}
			if(has_any_flags_of(events[i].events, EPOLLHUP | EPOLLERR)){
				int err_code;
				if(socket->did_time_out()){
					err_code = ETIMEDOUT;
				} else if(has_any_flags_of(events[i].events, EPOLLERR)){
					::socklen_t err_len = sizeof(err_code);
					if(::getsockopt(socket->get_fd(), SOL_SOCKET, SO_ERROR, &err_code, &err_len) != 0){
						err_code = errno;
						LOG_POSEIDON_WARNING("::getsockopt() failed: fd = ", socket->get_fd(), ", err_code = ", err_code, " (", get_error_desc(err_code), ")");
					}
				} else {
					err_code = 0;
				}
				LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_DEBUG, "Socket closed: remote = ", socket->get_remote_info(), ", typeid = ", typeid(*socket).name(), ", err_code = ", err_code, " (", get_error_desc(err_code), ")");
				g_socket_map.set_key<0, 3>(it, err_code);
			}
		}
		return true;
	}

	bool pump_one_readable_socket(boost::container::vector<unsigned char> &io_buffer) NOEXCEPT {
		PROFILE_ME;

		const AUTO(now, get_fast_mono_clock());
		boost::shared_ptr<SocketBase> socket;
		bool readable;
		{
			const RecursiveMutex::UniqueLock lock(g_mutex);
			const AUTO(it, g_socket_map.begin<1>());
			if(it == g_socket_map.end<1>()){
				return false;
			}
			if(now < it->read_time){
				return false;
			}
			socket = it->weakable->lock();
			if(!socket){
				g_socket_map.erase<1>(it);
				return true;
			}
			readable = it->readable;
		}

		if(socket->is_throttled()){
			LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_DEBUG, "Socket is throttled: socket = ", socket, ", typeid = ", typeid(*socket).name());
			const RecursiveMutex::UniqueLock lock(g_mutex);
			const AUTO(it, g_socket_map.find<0>(socket.get()));
			if(it != g_socket_map.end<0>()){
				g_socket_map.set_key<0, 1>(it, now + 5000);
			}
			return true;
		}

		int err_code;
		try {
			err_code = socket->poll_read_and_process(io_buffer.data(), io_buffer.size(), readable);
			if((err_code != 0) && (err_code != EINTR) && (err_code != EWOULDBLOCK) && (err_code != EAGAIN)){
				LOG_POSEIDON_DEBUG("Socket read error: socket = ", socket, ", typeid = ", typeid(*socket).name(), ", err_code = ", err_code, " (", get_error_desc(err_code), ")");
				socket->force_shutdown();
			}
		} catch(std::exception &e){
			LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_INFO, "std::exception thrown: what = ", e.what(), ", socket = ", socket, ", typeid = ", typeid(*socket).name());
			err_code = ECONNRESET;
			socket->force_shutdown();
		} catch(...){
			LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_INFO, "Unknown exception thrown: socket = ", socket, ", typeid = ", typeid(*socket).name());
			err_code = ECONNRESET;
			socket->force_shutdown();
		}
		if((err_code == EWOULDBLOCK) || (err_code == EAGAIN)){
			const RecursiveMutex::UniqueLock lock(g_mutex);
			const AUTO(it, g_socket_map.find<0>(socket.get()));
			if(it != g_socket_map.end<0>()){
				g_socket_map.set_key<0, 1>(it, (boost::uint64_t)-1);
			}
		}
		return true;
	}

	bool pump_one_writeable_socket(boost::container::vector<unsigned char> &io_buffer) NOEXCEPT {
		PROFILE_ME;

		const AUTO(now, get_fast_mono_clock());
		boost::shared_ptr<SocketBase> socket;
		bool writeable;
		{
			const RecursiveMutex::UniqueLock lock(g_mutex);
			const AUTO(it, g_socket_map.begin<2>());
			if(it == g_socket_map.end<2>()){
				return false;
			}
			if(now < it->write_time){
				return false;
			}
			socket = it->weakable->lock();
			if(!socket){
				g_socket_map.erase<2>(it);
				return true;
			}
			writeable = it->writeable;
		}

		Mutex::UniqueLock write_lock;
		int err_code;
		try {
			err_code = socket->poll_write(write_lock, io_buffer.data(), io_buffer.size(), writeable);
			if((err_code != 0) && (err_code != EINTR) && (err_code != EWOULDBLOCK) && (err_code != EAGAIN)){
				LOG_POSEIDON_DEBUG("Socket write error: socket = ", socket, ", typeid = ", typeid(*socket).name(), ", err_code = ", err_code, " (", get_error_desc(err_code), ")");
				socket->force_shutdown();
			}
		} catch(std::exception &e){
			LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_INFO, "std::exception thrown: what = ", e.what(), ", socket = ", socket, ", typeid = ", typeid(*socket).name());
			err_code = ECONNRESET;
			socket->force_shutdown();
		} catch(...){
			LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_INFO, "Unknown exception thrown: socket = ", socket, ", typeid = ", typeid(*socket).name());
			err_code = ECONNRESET;
			socket->force_shutdown();
		}
		if((err_code == EWOULDBLOCK) || (err_code == EAGAIN)){
			const RecursiveMutex::UniqueLock lock(g_mutex);
			const AUTO(it, g_socket_map.find<0>(socket.get()));
			if(it != g_socket_map.end<0>()){
				g_socket_map.set_key<0, 2>(it, (boost::uint64_t)-1);
			}
		}
		return true;
	}

	bool pump_one_closed_socket() NOEXCEPT {
		PROFILE_ME;

		// const AUTO(now, get_fast_mono_clock());
		boost::shared_ptr<SocketBase> socket;
		int err_code;
		{
			const RecursiveMutex::UniqueLock lock(g_mutex);
			const AUTO(it, g_socket_map.lower_bound<3>(0));
			if(it == g_socket_map.end<3>()){
				return false;
			}
			socket = it->weakable->lock();
			if(!socket){
				g_socket_map.erase<3>(it);
				return true;
			}
			err_code = it->err_code;
		}

		socket->mark_shutdown();
		try {
			socket->on_close(err_code);
		} catch(std::exception &e){
			LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_INFO, "std::exception thrown: what = ", e.what(), ", socket = ", socket, ", typeid = ", typeid(*socket).name());
		} catch(...){
			LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_INFO, "Unknown exception thrown: socket = ", socket, ", typeid = ", typeid(*socket).name());
		}
		{
			LOG_POSEIDON_DEBUG("Socket closed: socket = ", socket, ", typeid = ", typeid(*socket).name(), ", err_code = ", err_code, " (", get_error_desc(err_code), ")");
			const RecursiveMutex::UniqueLock lock(g_mutex);
			const AUTO(it, g_socket_map.find<0>(socket.get()));
			if(it != g_socket_map.end<0>()){
				g_socket_map.erase<0>(it);
			}
		}
		return true;
	}

	void thread_proc(){
		PROFILE_ME;
		LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_INFO, "Epoll daemon started.");

		boost::container::vector<unsigned char> io_buffer;
		const AUTO(io_buffer_size, MainConfig::get<std::size_t>("epoll_io_buffer_size", 4096));
		io_buffer.resize(std::max<std::size_t>(io_buffer_size, 508)); // 508 is the maximum size of UDP packets guaranteed to be transmitted.

		unsigned timeout = 0;
		for(;;){
			bool busy;
			do {
				busy = wait_for_sockets(0);
				busy += pump_one_readable_socket(io_buffer);
				busy += pump_one_writeable_socket(io_buffer);
				busy += pump_one_closed_socket();
				timeout = std::min(timeout * 2u + 1u, !busy * 100u);
			} while(busy);

			if(!atomic_load(g_running, ATOMIC_CONSUME)){
				break;
			}
			wait_for_sockets(timeout);
		}

		LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_INFO, "Epoll daemon stopped.");
	}
}

void EpollDaemon::start(){
	if(atomic_exchange(g_running, true, ATOMIC_ACQ_REL) != false){
		LOG_POSEIDON_FATAL("Only one daemon is allowed at the same time.");
		std::abort();
	}
	LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_INFO, "Starting epoll daemon...");

	DEBUG_THROW_UNLESS(g_epoll.reset(::epoll_create(100)), SystemException);
	Thread(&thread_proc, sslit("   N"), sslit("Network")).swap(g_thread);
}
void EpollDaemon::stop(){
	if(atomic_exchange(g_running, false, ATOMIC_ACQ_REL) == false){
		return;
	}
	LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_INFO, "Stopping epoll daemon...");

	if(g_thread.joinable()){
		g_thread.join();
	}

	const RecursiveMutex::UniqueLock lock(g_mutex);
	g_socket_map.clear();
	g_epoll.reset();
}

void EpollDaemon::add_socket(const boost::shared_ptr<SocketBase> &socket, bool take_ownership){
	PROFILE_ME;

	const AUTO(now, get_fast_mono_clock());
	const RecursiveMutex::UniqueLock lock(g_mutex);
	const AUTO(result, g_socket_map.insert(SocketElement(take_ownership, socket, now)));
	DEBUG_THROW_UNLESS(result.second, Exception, sslit("Socket is already in epoll"));
	try {
		::epoll_event event = { };
		event.events = static_cast<boost::uint32_t>(EPOLLIN | EPOLLOUT | EPOLLET);
		event.data.ptr = socket.get();
		DEBUG_THROW_UNLESS(::epoll_ctl(g_epoll.get(), EPOLL_CTL_ADD, socket->get_fd(), &event) == 0, SystemException);
	} catch(...){
		g_socket_map.erase(result.first);
		throw;
	}
}
bool EpollDaemon::mark_socket_writeable(const SocketBase *ptr) NOEXCEPT {
	PROFILE_ME;

	const RecursiveMutex::UniqueLock lock(g_mutex);
	const AUTO(it, g_socket_map.find<0>(ptr));
	if(it == g_socket_map.end()){
		LOG_POSEIDON_TRACE("Socket not found in epoll: ptr = ", ptr);
		return false;
	}
	const AUTO(now, get_fast_mono_clock());
	g_socket_map.set_key<0, 2>(it, now);
	return true;
}

void EpollDaemon::snapshot(boost::container::vector<EpollDaemon::SnapshotElement> &ret){
	PROFILE_ME;

	const RecursiveMutex::UniqueLock lock(g_mutex);
	ret.reserve(ret.size() + g_socket_map.size());
	for(AUTO(it, g_socket_map.begin()); it != g_socket_map.end(); ++it){
		const AUTO(socket, it->weakable->lock());
		if(!socket){
			continue;
		}
		SnapshotElement elem = { };
		elem.remote_info = socket->get_remote_info();
		elem.local_info = socket->get_local_info();
		elem.creation_time = socket->get_creation_time();
		elem.listening = socket->is_listening();
		elem.readable = it->readable;
		elem.writeable = it->writeable;
		ret.push_back(STD_MOVE(elem));
	}
}

}
