// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2015, LH_Mouse. All wrongs reserved.

#include "../precompiled.hpp"
#include "epoll_daemon.hpp"
#include "main_config.hpp"
#include "job_dispatcher.hpp"
#include "../thread.hpp"
#include "../epoll.hpp"
#include "../log.hpp"
#include "../atomic.hpp"
#include "../time.hpp"
#include "../tcp_session_base.hpp"
#include "../socket_server_base.hpp"
#include "../profiler.hpp"

namespace Poseidon {

namespace {
	std::size_t g_max_timeout               = 100;
	boost::uint64_t g_tcp_request_timeout       = 30000;

	volatile bool g_running = false;
	Thread g_thread;

	boost::shared_ptr<Epoll> g_epoll = boost::make_shared<Epoll>();

	Mutex g_server_mutex;
	std::vector<boost::weak_ptr<const SocketServerBase> > g_servers;

	std::size_t poll_servers(){
		std::vector<boost::shared_ptr<const SocketServerBase> > servers;
		{
			const Mutex::UniqueLock lock(g_server_mutex);
			servers.reserve(g_servers.size());
			AUTO(it, g_servers.begin());
			while(it != g_servers.end()){
				AUTO(server, it->lock());
				if(!server){
					it = g_servers.erase(it);
					continue;
				}
				servers.push_back(STD_MOVE(server));
				++it;
			}
		}
		std::size_t count = 0;
		for(AUTO(it, servers.begin()); it != servers.end(); ++it){
			try {
				if(!(*it)->poll()){
					continue;
				}
				++count;
			} catch(std::exception &e){
				LOG_POSEIDON_WARNING("std::exception thrown while accepting connection: what = ", e.what());
			} catch(...){
				LOG_POSEIDON_WARNING("Unknown exception thrown while accepting connection.");
			}
		}
		return count;
	}

	void daemon_loop(){
		boost::uint64_t epoll_timeout = 0;
		for(;;){
			bool busy = false;

			try {
				if(JobDispatcher::is_running()){
					if(poll_servers() > 0){
						++busy;
					}
					if(g_epoll->pump_readable() > 0){
						++busy;
					}
				}
				if(g_epoll->pump_writeable() > 0){
					++busy;
				}
				if(g_epoll->wait(epoll_timeout) > 0){
					++busy;
				}
				// 二次指数回退算法。如果有连接接入（忙），epoll 等待时间就短一些；反之（闲）亦然。
				if(busy){
					epoll_timeout = 0;
				} else {
					epoll_timeout |= 1;
					epoll_timeout <<= 1;
					if(epoll_timeout > g_max_timeout){
						epoll_timeout = g_max_timeout;
					}
				}
			} catch(std::exception &e){
				LOG_POSEIDON_ERROR("std::exception thrown while flush data: what = ", e.what());
			} catch(...){
				LOG_POSEIDON_ERROR("Unknown exception thrown while flush data.");
			}

			if(!busy && !atomic_load(g_running, ATOMIC_CONSUME)){
				break;
			}
		}
	}

	void thread_proc(){
		PROFILE_ME;
		LOG_POSEIDON_INFO("Epoll daemon started.");

		daemon_loop();

		LOG_POSEIDON_INFO("Epoll daemon stopped.");
	}
}

void EpollDaemon::start(){
	if(atomic_exchange(g_running, true, ATOMIC_ACQ_REL) != false){
		LOG_POSEIDON_FATAL("Only one daemon is allowed at the same time.");
		std::abort();
	}
	LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_INFO, "Starting epoll daemon...");

	MainConfig::get(g_max_timeout, "epoll_max_timeout");
	LOG_POSEIDON_DEBUG("Max timeout = ", g_max_timeout);

	MainConfig::get(g_tcp_request_timeout, "epoll_tcp_request_timeout");
	LOG_POSEIDON_DEBUG("Tcp request timeout = ", g_tcp_request_timeout);

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
	g_epoll->clear();
	g_servers.clear();
}

boost::uint64_t EpollDaemon::get_tcp_request_timeout(){
	return g_tcp_request_timeout;
}

std::vector<EpollDaemon::SnapshotElement> EpollDaemon::snapshot(){
	std::vector<boost::shared_ptr<TcpSessionBase> > sessions;
	g_epoll->snapshot(sessions);

	std::vector<SnapshotElement> ret;
	const AUTO(now, get_fast_mono_clock());
	for(AUTO(it, sessions.begin()); it != sessions.end(); ++it){
		ret.push_back(SnapshotElement());
		AUTO_REF(item, ret.back());
		item.remote = (*it)->get_remote_info();
		item.local = (*it)->get_local_info();
		item.ms_online = now - (*it)->get_created_time();
	}
	return ret;
}

void EpollDaemon::add_session(const boost::shared_ptr<TcpSessionBase> &session){
	g_epoll->add_session(session);
}
void EpollDaemon::register_server(boost::weak_ptr<const SocketServerBase> server){
	const Mutex::UniqueLock lock(g_server_mutex);
	g_servers.push_back(STD_MOVE(server));
}

}
