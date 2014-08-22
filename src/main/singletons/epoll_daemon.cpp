#include "../../precompiled.hpp"
#include "epoll_daemon.hpp"
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

void addSocket(const boost::shared_ptr<TcpPeer> &peer){
	::epoll_event event;
	event.events = EPOLLHUP | EPOLLERR | EPOLLIN | EPOLLOUT;
    event.data.ptr = peer.get();

	const boost::mutex::scoped_lock lock(g_mutex);
	AUTO(const result, g_epollSockets.insert(peer));
	if(result.second){
		if(::epoll_ctl(g_epoll.get(), EPOLL_CTL_ADD, peer->getFd(), &event) != 0){
			g_epollSockets.erase(result.first);
			DEBUG_THROW(SystemError, errno);
		}
	} else {
		if(::epoll_ctl(g_epoll.get(), EPOLL_CTL_MOD, peer->getFd(), &event) != 0){
			DEBUG_THROW(SystemError, errno);
		}
	}
}
void removeSocket(const boost::shared_ptr<TcpPeer> &peer) throw() {
	const boost::mutex::scoped_lock lock(g_mutex);
	AUTO(const it, g_epollSockets.find(peer));
	if(it == g_epollSockets.end()){
		LOG_WARNING <<"Trying to remove a non-existent socket.";
		return;
	}
	g_epollSockets.erase(it);
	if(::epoll_ctl(g_epoll.get(), EPOLL_CTL_DEL, peer->getFd(), NULL) != 0){
		const int code = errno;
		LOG_WARNING <<"epoll_ctl() failed, errno = " <<code;
	}
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
			boost::shared_ptr<TcpPeer> peer = ((TcpPeer *)event.data.ptr)->shared_from_this();
			try {
				if(event.events & EPOLLHUP){
					LOG_INFO <<"Socket has been hung up. Remove it.";
					removeSocket(peer);
					continue;
				}
				if(event.events & EPOLLERR){
					int err;
					::socklen_t errLen = sizeof(err);
					if(::getsockopt(g_epoll.get(), SOL_SOCKET, SO_ERROR, &err, &errLen) != 0){
						err = errno;
					}
					DEBUG_THROW(SystemError, err);
				}
				if(event.events & EPOLLIN){
					unsigned char data[4096];
					// 如果我们用 edge-triggered epoll 这个地方得改。
					const ::ssize_t bytesRead = ::recv(peer->getFd(), data, sizeof(data), 0);
					if(bytesRead < 0){
						if((errno != EAGAIN) && (errno != EINTR)){
							DEBUG_THROW(SystemError, errno);
						}
					} else if(bytesRead == 0){
						LOG_INFO <<"Socket has been closed by peer. Remove it.";
						removeSocket(peer);
					} else {
						// 忽略 EAGAIN 和 EINTR，此时 bytesRead 为 -1。
						if(bytesRead > 0){
							peer->onReadAvail(data, bytesRead);
						}
					}
				}
				if(event.events & EPOLLOUT){
					unsigned char data[4096];
					std::size_t toWrite = 0;
					for(;;){
						toWrite = peer->peekWriteAvail(data, sizeof(data));
						if(toWrite == 0){
							break;
						}
						const ::ssize_t bytesWritten = ::send(peer->getFd(), data, toWrite, 0);
						if(bytesWritten < 0){
							if((errno != EAGAIN) && (errno != EINTR)){
								DEBUG_THROW(SystemError, errno);
							}
							break;
						} else if(bytesWritten == 0){
							break;
						}
						peer->notifyWritten(bytesWritten);
					}
					if(toWrite == 0){
						if(peer->hasBeenShutdown()){
							::shutdown(peer->getFd(), SHUT_RDWR);
						}
					} else {
						// EPOLLOUT 只在两种条件下才出现：
						// 调用 epoll_ctl 后会返回一次，EAGAIN 返回后重新变为可写状态也会返回一次。
						// 如果当前的写入操作没有完成，我们需要告诉 epoll 等到下次循环时重试。
						::epoll_ctl(g_epoll.get(), EPOLL_CTL_MOD, peer->getFd(), &event);
					}
				}
			} catch(Exception &e){
				LOG_ERROR <<"Exception thrown while dispatching data: file = "
					<<e.file() <<", line = " <<e.line() <<", what = " <<e.what();
				removeSocket(peer);
			} catch(std::exception &e){
				LOG_ERROR <<"std::exception thrown while dispatching data: what = " <<e.what();
				removeSocket(peer);
			} catch(...){
				LOG_ERROR <<"Unknown exception thrown while dispatching data.";
				removeSocket(peer);
			}
		}
		if(ready > 0){
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

void EpollDaemon::start(){
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
void EpollDaemon::stop(){
	LOG_INFO <<"Stopping epoll daemon...";

	atomicStore(g_daemonRunning, false);
	if(g_daemonThread.joinable()){
		g_daemonThread.join();
	}
	g_epollSockets.clear();
	g_idleCallbacks.clear();
}

void EpollDaemon::registerTcpPeer(const boost::shared_ptr<TcpPeer> &readable){
	addSocket(readable);
}
void EpollDaemon::pendWrite(const boost::shared_ptr<TcpPeer> &writeable){
	addSocket(writeable);
}
void EpollDaemon::registerIdleCallback(boost::function<bool ()> callback){
	const boost::mutex::scoped_lock lock(g_mutex);
	g_idleCallbacks.push_back(callback);
}
