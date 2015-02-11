// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2015, LH_Mouse. All wrongs reserved.

#include "../precompiled.hpp"
#include "epoll_daemon.hpp"
#include "main_config.hpp"
#include <boost/thread.hpp>
#include "../epoll.hpp"
#include "../log.hpp"
#include "../atomic.hpp"
#include "../utilities.hpp"
#include "../tcp_session_base.hpp"
#include "../socket_server_base.hpp"
#include "../profiler.hpp"
using namespace Poseidon;

namespace {

std::size_t	g_maxTimeout				= 100;
boost::uint64_t g_tcpRequestTimeout		= 30000;

volatile bool g_running = false;
boost::thread g_thread;

Epoll g_epoll;

boost::mutex g_serverMutex;
std::list<boost::weak_ptr<const SocketServerBase> > g_servers;

std::size_t pollServers(){
	std::vector<boost::shared_ptr<const SocketServerBase> > servers;
	{
		const boost::mutex::scoped_lock lock(g_serverMutex);
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
			LOG_POSEIDON_ERROR("std::exception thrown while accepting connection: what = ", e.what());
		} catch(...){
			LOG_POSEIDON_ERROR("Unknown exception thrown while accepting connection.");
		}
	}
	return count;
}

void daemonLoop(){
	std::size_t epollTimeout = 0;
	while(atomicLoad(g_running, ATOMIC_ACQUIRE)){
		try {
			bool busy = false;
			if(g_epoll.wait(epollTimeout) > 0){
				++busy;
			}
			if(g_epoll.pumpReadable() > 0){
				++busy;
			}
			if(g_epoll.pumpWriteable() > 0){
				++busy;
			}
			if(pollServers() > 0){
				++busy;
			}
			// 二次指数回退算法。如果有连接接入（忙），epoll 等待时间就短一些；反之（闲）亦然。
			if(busy){
				epollTimeout = 0;
			} else {
				epollTimeout |= 1;
				epollTimeout <<= 1;
				if(epollTimeout > g_maxTimeout){
					epollTimeout = g_maxTimeout;
				}
			}
		} catch(std::exception &e){
			LOG_POSEIDON_ERROR("std::exception thrown while flush data: what = ", e.what());
		} catch(...){
			LOG_POSEIDON_ERROR("Unknown exception thrown while flush data.");
		}
	}
	while(g_epoll.pumpWriteable() > 0){
		// noop
	}
}

void threadProc(){
	PROFILE_ME;
	Logger::setThreadTag("   N"); // Network
	LOG_POSEIDON_INFO("Epoll daemon started.");

	daemonLoop();

	LOG_POSEIDON_INFO("Epoll daemon stopped.");
}

}

void EpollDaemon::start(){
	if(atomicExchange(g_running, true, ATOMIC_ACQ_REL) != false){
		LOG_POSEIDON_FATAL("Only one daemon is allowed at the same time.");
		std::abort();
	}
	LOG_POSEIDON_INFO("Starting epoll daemon...");

	AUTO_REF(conf, MainConfig::getConfigFile());

	conf.get(g_maxTimeout, "epoll_max_timeout");
	LOG_POSEIDON_DEBUG("Max timeout = ", g_maxTimeout);

	conf.get(g_tcpRequestTimeout, "epoll_tcp_request_timeout");
	LOG_POSEIDON_DEBUG("Tcp request timeout = ", g_tcpRequestTimeout);

	boost::thread(threadProc).swap(g_thread);
}
void EpollDaemon::stop(){
	if(atomicExchange(g_running, false, ATOMIC_ACQ_REL) == false){
		return;
	}
	LOG_POSEIDON_INFO("Stopping epoll daemon...");

	if(g_thread.joinable()){
		g_thread.join();
	}
	g_epoll.clear();
	g_servers.clear();
}

boost::uint64_t EpollDaemon::getTcpRequestTimeout(){
	return g_tcpRequestTimeout;
}

std::vector<EpollSnapshotItem> EpollDaemon::snapshot(){
	std::vector<boost::shared_ptr<TcpSessionBase> > sessions;
	g_epoll.snapshot(sessions);

	std::vector<EpollSnapshotItem> ret;
	const AUTO(now, getFastMonoClock());
	for(AUTO(it, sessions.begin()); it != sessions.end(); ++it){
		ret.push_back(EpollSnapshotItem());
		AUTO_REF(item, ret.back());
		item.remote = (*it)->getRemoteInfo();
		item.local = (*it)->getLocalInfo();
		item.msOnline = now - (*it)->getCreatedTime();
	}
	return ret;
}

void EpollDaemon::addSession(const boost::shared_ptr<TcpSessionBase> &session){
	if(session->hasBeenShutdown()){
		return;
	}
	g_epoll.addSession(session);
}

void EpollDaemon::registerServer(boost::weak_ptr<const SocketServerBase> server){
	const boost::mutex::scoped_lock lock(g_serverMutex);
	g_servers.push_back(STD_MOVE(server));
}
