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
#include "../utilities.hpp"
#include "../exception.hpp"
#include "../tcp_peer.hpp"
using namespace Poseidon;

namespace {

volatile bool g_daemonRunning = false;
ScopedFile g_epoll;
boost::thread g_daemonThread;

boost::mutex g_readableMutex;
std::set<boost::shared_ptr<TcpPeer> > g_readable;

boost::mutex g_writeableMutex;
std::set<boost::shared_ptr<TcpPeer> > g_writeable;

boost::mutex g_callbackMutex;
std::list<boost::function<bool ()> > g_idleCallbacks;

void addReadable(const boost::shared_ptr<TcpPeer> &peer){
	::epoll_event event;
	event.events = EPOLLHUP | EPOLLERR | EPOLLIN | EPOLLET;
	event.data.ptr = peer.get();

	const boost::mutex::scoped_lock lock(g_readableMutex);
	AUTO(const result, g_readable.insert(peer));
	if(result.second){
		if(::epoll_ctl(g_epoll.get(), EPOLL_CTL_ADD, peer->getFd(), &event) != 0){
			g_readable.erase(result.first);
			DEBUG_THROW(SystemError, errno);
		}
	}
}
void removePeer(const boost::shared_ptr<TcpPeer> &peer){
	{
		const boost::mutex::scoped_lock lock(g_readableMutex);
		AUTO(const it, g_readable.find(peer));
		if(it == g_readable.end()){
			LOG_WARNING, "Attempting to remove a non-existent socket.";
			return;
		}
		g_readable.erase(it);
	}
	if(::epoll_ctl(g_epoll.get(), EPOLL_CTL_DEL, peer->getFd(), NULL) != 0){
		AUTO(const desc, getErrorDesc());
		LOG_WARNING, "::epoll_ctl() failed: ", desc;
	}
}

void threadProc(){
	LOG_INFO, "Epoll daemon started.";

	unsigned epollTimeout = 1;
	std::vector<boost::shared_ptr<TcpPeer> > peers;
	while(atomicLoad(g_daemonRunning)){
		// 第一部分，发送数据。
		{
			peers.clear();
			const boost::mutex::scoped_lock lock(g_writeableMutex);
			AUTO(it, g_writeable.begin());
			while(it != g_writeable.end()){
				if((*it)->hasBeenShutdown() || ((*it)->peekWriteAvail(NULL, 0) == 0)){
					g_writeable.erase(it++);
				} else {
					peers.push_back(*it);
					++it;
				}
			}
		}
		for(AUTO(it, peers.begin()); it != peers.end(); ++it){
			AUTO_REF(peer, *it);
			unsigned char data[1024];
			const std::size_t avail = peer->peekWriteAvail(data, sizeof(data));
			if(avail == 0){
				continue;
			}
			const ::ssize_t bytesWritten = ::send(peer->getFd(), data, avail, 0);
			if(bytesWritten < 0){
				if((errno == EINTR) || (errno == EAGAIN)){
					continue;
				}
				AUTO(const desc, getErrorDesc(errno));
				LOG_WARNING, "Write error in socket: ", desc;
				continue;
			}
			peer->notifyWritten(bytesWritten);
		}
		// 二次指数回退算法，如果有数据可写就等得时间短一些，反之亦然。
		if(peers.empty()){
			if(epollTimeout < 0x100){
				epollTimeout <<= 1;
			}
		} else {
			epollTimeout = 1;
		}

		// 第二部分，接收数据。
		::epoll_event events[64];
		const int ready = ::epoll_wait(g_epoll.get(), events, COUNT_OF(events), epollTimeout);
		if(ready < 0){
			AUTO(const desc, getErrorDesc());
			LOG_ERROR, "::epoll_wait() failed: ", desc;
		} else for(std::size_t i = 0; i < (unsigned)ready; ++i){
			epoll_event &event = events[i];
			AUTO(const peer, static_cast<TcpPeer *>(event.data.ptr)->virtualSharedFromThis<TcpPeer>());
			if(event.events & EPOLLHUP){
				LOG_INFO, "Socket has been hung up. Remove it.";
				removePeer(peer);
				continue;
			}
			if(event.events & EPOLLERR){
				int err;
				::socklen_t errLen = sizeof(err);
				if(::getsockopt(g_epoll.get(), SOL_SOCKET, SO_ERROR, &err, &errLen) != 0){
					err = errno;
				}
				AUTO(const desc, getErrorDesc(err));
				LOG_WARNING, "Error in socket: ", desc;
				removePeer(peer);
				continue;
			}
			if(event.events & EPOLLIN){
				bool toRemove = true;
				for(;;){
					unsigned char data[1024];
					const ::ssize_t bytesRead = ::recv(peer->getFd(), data, sizeof(data), 0);
					if(bytesRead < 0){
						if(errno == EINTR){
							continue;
						}
						if(errno == EAGAIN){
							toRemove = false;
							break;
						}
						AUTO(const desc, getErrorDesc(errno));
						LOG_WARNING, "Read error in socket: ", desc;
						break;
					} else if(bytesRead == 0){
						LOG_INFO, "Socket has been closed by peer. Remove it.";
						break;
					}
					try {
						peer->onReadAvail(data, bytesRead);
					} catch(Exception &e){
						LOG_ERROR, "Exception thrown while dispatching data: file = ", e.file(),
							", line = ", e.line(), ", what = ", e.what();
						break;
					} catch(std::exception &e){
						LOG_ERROR, "std::exception thrown while dispatching data: what = ", e.what();
						break;
					} catch(...){
						LOG_ERROR, "Unknown exception thrown while dispatching data.";
						break;
					}
				}
				if(toRemove){
					removePeer(peer);
					continue;
				}
			}
		}

		// 第三部分，处理空闲时回调。
		boost::mutex::scoped_lock lock(g_callbackMutex);
		AUTO(it, g_idleCallbacks.begin());
		while(it != g_idleCallbacks.end()){
			lock.unlock();

			// list 具有随机插入删除而迭代器不失效的特性，理解这一点非常重要。
			bool result = true;
			try {
				result = (*it)();
			} catch(Exception &e){
				LOG_ERROR, "Exception thrown while executing an idle callback, file = ", e.file(),
					", line = ", e.line(), ": what = ", e.what();
			} catch(std::exception &e){
				LOG_ERROR, "std::exception thrown while executing an idle callback: what = ", e.what();
			} catch(...){
				LOG_ERROR, "Unknown exception thrown while dispatching data.";
			}

			lock.lock();
			if(result){
				++it;
			} else {
				LOG_DEBUG, "An idle callback returned false, remove it.";
				it = g_idleCallbacks.erase(it);
			}
		}
	}

	LOG_INFO, "Epoll daemon stopped.";
}

}

void EpollDaemon::start(){
	if(atomicExchange(g_daemonRunning, true) != false){
		LOG_FATAL, "Only one daemon is allowed at the same time.";
		std::abort();
	}
	LOG_INFO, "Starting epoll daemon...";

	g_epoll.reset(::epoll_create(4096));
	if(!g_epoll){
		DEBUG_THROW(SystemError, errno);
	}
	boost::thread(threadProc).swap(g_daemonThread);
}
void EpollDaemon::stop(){
	LOG_INFO, "Stopping epoll daemon...";

	atomicStore(g_daemonRunning, false);
	if(g_daemonThread.joinable()){
		g_daemonThread.join();
	}
	g_readable.clear();
	g_writeable.clear();
	g_idleCallbacks.clear();
}

void EpollDaemon::notifyReadable(boost::shared_ptr<TcpPeer> peer){
	assert(peer);

	addReadable(peer);
}
void EpollDaemon::notifyWriteable(boost::shared_ptr<TcpPeer> peer){
	assert(peer);

	const boost::mutex::scoped_lock lock(g_writeableMutex);
	g_writeable.insert(peer);
}
void EpollDaemon::registerIdleCallback(boost::function<bool ()> callback){
	const boost::mutex::scoped_lock lock(g_callbackMutex);
	g_idleCallbacks.push_back(callback);
}
