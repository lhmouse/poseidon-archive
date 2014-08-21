#include "../../precompiled.hpp"
#include "session_manager.hpp"
#include <set>
#include <list>
#include <boost/thread.hpp>
#include <boost/make_shared.hpp>
#include <sys/epoll.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <errno.h>
#include <string.h>
#include "../log.hpp"
#include "../atomic.hpp"
#include "../exception.hpp"
#include "../tcp_peer.hpp"
using namespace Poseidon;

namespace {

volatile bool g_daemonRunning = false;
ScopedFile g_epoll;
boost::thread g_daemonThread;

boost::mutex g_mutex;
std::set<boost::shared_ptr<TcpPeer> > g_epollSockets;
std::list<boost::function<bool ()> > g_idleCallbacks;

void removeSocket(TcpPeer &peer) throw() {
	::epoll_event ignored; // epoll_ctl BUGS.
	::epoll_ctl(g_epoll.get(), EPOLL_CTL_DEL, peer.getFd(), &ignored);
	// 下面这行是不分配内存也不会抛出异常的。
	boost::shared_ptr<TcpPeer> toFind(boost::shared_ptr<void>(), &peer);
	const boost::mutex::scoped_lock lock(g_mutex);
	g_epollSockets.erase(toFind);
}

void threadProc() throw() {
	LOG_INFO <<"epoll daemon started.";

	unsigned epollTimeout = 1;
	while(atomicLoad(g_daemonRunning)){
		// 第一部分，转发一些套接字数据。
		::epoll_event events[64];
		const int ready = ::epoll_wait(g_epoll.get(), events, COUNT_OF(events), epollTimeout);
		if(ready < 0){
			char temp[256];
			const char *const desc = ::strerror_r(errno, temp, sizeof(temp));
			LOG_ERROR <<"::epoll_wait() has failed: " <<desc;
		} else for(std::size_t i = 0; i < (unsigned)ready; ++i){
			epoll_event &event = events[i];
			TcpPeer &peer = *(TcpPeer *)event.data.ptr;

			if(event.events | EPOLLERR){
				LOG_INFO <<"epoll indicates there is an error in socket " <<&peer <<", remove it";

				removeSocket(peer);
				continue;
			}
			if(event.events | EPOLLIN){
				unsigned char data[4096];
				const ssize_t size = ::recv(peer.getFd(), data, sizeof(data), 0);
				LOG_DEBUG <<"Read " <<size <<" bytes from socket " <<&peer <<"...";

				if(size < 0){
					char temp[256];
					const char *const desc = ::strerror_r(errno, temp, sizeof(temp));
					LOG_ERROR <<"Read error: " <<desc;
					removeSocket(peer);
					continue;
				}
				if(size == 0){
					LOG_DEBUG <<"Socket was shutdown gracefully";
					removeSocket(peer);
					continue;
				}
				try {
					peer.onDataAvail(data, size);
				} catch(Exception &e){
					LOG_ERROR <<"Exception thrown while dispatching data: file = "
						<<e.file() <<", line = " <<e.line() <<", what = " <<e.what();
				} catch(std::exception &e){
					LOG_ERROR <<"std::exception thrown while dispatching data: what = " <<e.what();
				} catch(...){
					LOG_ERROR <<"Unknown exception thrown while dispatching data.";
				}
			}
		}
		if(ready != 0){
			epollTimeout = 1;
		} else if(epollTimeout < 0x100){
			// 指数回退算法。如果数据包多一些等待时间就短一些，反之亦然。
			epollTimeout <<= 1;
		}

		// 第二部分，处理空闲时回调。
		boost::mutex::scoped_lock lock(g_mutex);
		AUTO(it, g_idleCallbacks.begin());
		while(it != g_idleCallbacks.end()){
			lock.unlock();
			// list 具有随机插入而迭代器不失效的特性，理解这一点非常重要。
			bool result = true;
			try {
				result = (*it)();
			} catch(Exception &e){
				LOG_ERROR <<"Exception thrown while executing an idle callback, file = "
					<<e.file() <<", line = " <<e.line() <<": what = " <<e.what();
			} catch(std::exception &e){
				LOG_ERROR <<"std::exception thrown while executing an idle callback: what = "
					<<e.what();
			} catch(...){
				LOG_ERROR <<"Unknown exception thrown while dispatching data.";
			}
			lock.lock();
			if(result){
				++it;
			} else {
				LOG_DEBUG <<"An idle callback returned false, remove it.";
				it = g_idleCallbacks.erase(it);
			}
		}
	}

	LOG_INFO <<"epoll daemon stopped.";
}

}

void SessionManager::startDaemon(){
	if(atomicExchange(g_daemonRunning, true) != false){
		LOG_FATAL <<"Only one daemon is allowed at the same time.";
		std::abort();
	}
	LOG_INFO <<"Starting epoll daemon...";

	g_epoll.reset(::epoll_create(4096));
	if(!g_epoll){
		DEBUG_THROW(SystemError, errno);
	}
	boost::thread(threadProc).swap(g_daemonThread);
}
void SessionManager::stopDaemon(){
	LOG_INFO <<"Stopping epoll daemon...";

	atomicStore(g_daemonRunning, false);
	if(g_daemonThread.joinable()){
		g_daemonThread.join();
	}
}

void SessionManager::registerTcpPeer(boost::shared_ptr<TcpPeer> readable){
	const boost::mutex::scoped_lock lock(g_mutex);
	g_epollSockets.insert(readable);
}
void SessionManager::registerIdleCallback(boost::function<bool ()> callback){
	const boost::mutex::scoped_lock lock(g_mutex);
	g_idleCallbacks.push_back(callback);
}
