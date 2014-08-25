#include "../../precompiled.hpp"
#include "epoll_daemon.hpp"
#include <set>
#include <boost/thread.hpp>
#include <boost/make_shared.hpp>
#include <sys/epoll.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <errno.h>
#include <string.h>
#include <csignal>
#include "../log.hpp"
#include "../atomic.hpp"
#include "../utilities.hpp"
#include "../exception.hpp"
#include "../tcp_peer.hpp"
#include "../socket_server_base.hpp"
using namespace Poseidon;

namespace {

volatile bool g_daemonRunning = false;

// 读线程。
ScopedFile g_readerEpoll;
boost::thread g_readerThread;

std::set<boost::shared_ptr<TcpPeer> > g_readable;

boost::mutex g_serverMutex;
std::set<boost::shared_ptr<const SocketServerBase> > g_servers;

int addReadable(const boost::shared_ptr<TcpPeer> &peer){
	AUTO(const result, g_readable.insert(peer));
	if(result.second){
		::epoll_event event;
		event.events = EPOLLHUP | EPOLLERR | EPOLLIN;
		event.data.ptr = peer.get();
		if(::epoll_ctl(g_readerEpoll.get(), EPOLL_CTL_ADD, peer->getFd(), &event) != 0){
			g_readable.erase(result.first);
			return errno;
		}
	}
	return 0;
}
void removeReadable(const boost::shared_ptr<TcpPeer> &peer){
	AUTO(const it, g_readable.find(peer));
	if(it == g_readable.end()){
		return;
	}
	g_readable.erase(it);
	if(::epoll_ctl(g_readerEpoll.get(), EPOLL_CTL_DEL, peer->getFd(), NULL)){
		AUTO(const desc, getErrorDesc());
		LOG_WARNING, "Epoll failed to remove socket: ", desc;
	}
}

bool readerLoop(unsigned timeout){
	bool notIdle = false;

	::epoll_event events[256];
	const int ready = ::epoll_wait(g_readerEpoll.get(), events, COUNT_OF(events), timeout);
	if(ready < 0){
		AUTO(const desc, getErrorDesc());
		LOG_ERROR, "::epoll_wait() failed: ", desc;
		return false;
	}
	for(unsigned i = 0; i < (unsigned)ready; ++i){
		::epoll_event &event = events[i];
		AUTO(const peer, static_cast<TcpPeer *>(event.data.ptr)->virtualSharedFromThis<TcpPeer>());
		try {
			if(event.events & EPOLLHUP){
				LOG_INFO, "Socket has been hung up. Remove it.";
				EpollDaemon::resetPeer(peer);
				continue;
			}
			if(event.events & EPOLLERR){
				int err;
				::socklen_t errLen = sizeof(err);
				if(::getsockopt(peer->getFd(), SOL_SOCKET, SO_ERROR, &err, &errLen) != 0){
					err = errno;
				}
				DEBUG_THROW(SystemError, err);
			}
			if(event.events & EPOLLIN){
				notIdle = true;
				unsigned char data[1024];
				const ::ssize_t bytesRead = ::recv(peer->getFd(), data, sizeof(data), MSG_NOSIGNAL);
				if(bytesRead < 0){
					if((errno == EINTR) || (errno == EAGAIN)){
						continue;
					}
					DEBUG_THROW(SystemError, errno);
				} else if(bytesRead == 0){
					LOG_INFO, "Socket has been closed by peer. Remove it.";
					removeReadable(peer);
					continue;
				}
				peer->onReadAvail(data, bytesRead);
			}
		} catch(Exception &e){
			LOG_ERROR, "Exception thrown while dispatching data: file = ", e.file(),
				", line = ", e.line(), ", what = ", e.what();
			EpollDaemon::resetPeer(peer);
		} catch(std::exception &e){
			LOG_ERROR, "std::exception thrown while dispatching data: what = ", e.what();
			EpollDaemon::resetPeer(peer);
		} catch(...){
			LOG_ERROR, "Unknown exception thrown while dispatching data.";
			EpollDaemon::resetPeer(peer);
		}
	}

	std::vector<boost::shared_ptr<const SocketServerBase> > servers;
	servers.reserve(16);
	{
		const boost::mutex::scoped_lock lock(g_serverMutex);
		std::copy(g_servers.begin(), g_servers.end(), std::back_inserter(servers));
	}
	for(AUTO(it, servers.begin()); it != servers.end(); ++it){
		try {
			AUTO(peer, (*it)->tryAccept());
			if(!peer){
				continue;
			}
			notIdle = true;
			addReadable(peer);
		} catch(Exception &e){
			LOG_ERROR, "Exception thrown while accepting client: file = ", e.file(),
				", line = ", e.line(), ", what = ", e.what();
		} catch(std::exception &e){
			LOG_ERROR, "std::exception thrown while accepting client: what = ", e.what();
		} catch(...){
			LOG_ERROR, "Unknown exception thrown while accepting client.";
		}
	}

	return notIdle;
}

// 写线程。
ScopedFile g_writerEpoll;
boost::thread g_writerThread;

boost::mutex g_writeableMutex;
std::set<boost::shared_ptr<TcpPeer> > g_writeable;

int addWritable(const boost::shared_ptr<TcpPeer> &peer){
	const boost::mutex::scoped_lock lock(g_writeableMutex);
	AUTO(const result, g_writeable.insert(peer));
	if(result.second){
		::epoll_event event;
		event.events = EPOLLHUP | EPOLLERR | EPOLLOUT;
		event.data.ptr = peer.get();
		if(::epoll_ctl(g_writerEpoll.get(), EPOLL_CTL_ADD, peer->getFd(), &event) != 0){
			g_writeable.erase(result.first);
			return errno;
		}
	}
	return 0;
}
void removeWriteable(const boost::shared_ptr<TcpPeer> &peer){
	const boost::mutex::scoped_lock lock(g_writeableMutex);
	AUTO(const it, g_writeable.find(peer));
	if(it == g_writeable.end()){
		return;
	}
	g_writeable.erase(it);
	if(::epoll_ctl(g_writerEpoll.get(), EPOLL_CTL_DEL, peer->getFd(), NULL) != 0){
		AUTO(const desc, getErrorDesc());
		LOG_WARNING, "Epoll failed to remove socket: ", desc;
	}
}

bool writerLoop(unsigned timeout){
	bool notIdle = false;

	::epoll_event events[256];
	const int ready = ::epoll_wait(g_writerEpoll.get(), events, COUNT_OF(events), timeout);
	if(ready < 0){
		AUTO(const desc, getErrorDesc());
		LOG_ERROR, "::epoll_wait() failed: ", desc;
		return false;
	}
	for(unsigned i = 0; i < (unsigned)ready; ++i){
		::epoll_event &event = events[i];
		AUTO(const peer, static_cast<TcpPeer *>(event.data.ptr)->virtualSharedFromThis<TcpPeer>());
		try {
			if(event.events & EPOLLHUP){
				LOG_INFO, "Socket has been hung up. Remove it.";
				EpollDaemon::resetPeer(peer);
				continue;
			}
			if(event.events & EPOLLERR){
				int err;
				::socklen_t errLen = sizeof(err);
				if(::getsockopt(peer->getFd(), SOL_SOCKET, SO_ERROR, &err, &errLen) != 0){
					err = errno;
				}
				DEBUG_THROW(SystemError, err);
			}
			if(event.events & EPOLLOUT){
				notIdle = true;
				unsigned char data[1024];
				std::size_t bytesToWrite;
				{
					boost::mutex::scoped_lock lock;
					bytesToWrite = peer->peekWriteAvail(lock, data, sizeof(data));
					if(bytesToWrite == 0){
						removeWriteable(peer);
						if(peer->hasBeenShutdown()){
							peer->forceShutdown();
						}
						continue;
					}
				}
				const ::ssize_t bytesWritten = ::send(peer->getFd(), data, bytesToWrite, MSG_NOSIGNAL);
				if(bytesWritten < 0){
					if((errno == EINTR) || (errno == EAGAIN)){
						continue;
					}
					DEBUG_THROW(SystemError, errno);
				} else if(bytesWritten == 0){
					continue;
				}
				peer->notifyWritten(bytesWritten);
			}
		} catch(Exception &e){
			LOG_ERROR, "Exception thrown while writing socket: file = ", e.file(),
				", line = ", e.line(), ", what = ", e.what();
			EpollDaemon::resetPeer(peer);
		} catch(std::exception &e){
			LOG_ERROR, "std::exception thrown while writing socket: what = ", e.what();
			EpollDaemon::resetPeer(peer);
		} catch(...){
			LOG_ERROR, "Unknown exception thrown while writing socket.";
			EpollDaemon::resetPeer(peer);
		}
	}

	return notIdle;
}

// 线程起始函数。
void readerThreadProc(){
	LOG_INFO, "Epoll reader thread started.";

	std::signal(SIGPIPE, SIG_IGN);

	// 二次指数回退算法，如果无事可做就等得时间长一些，反之亦然。
	unsigned timeout = 1;
	while(atomicLoad(g_daemonRunning)){
		if(readerLoop(timeout)){
			timeout = 1;
		} else if(timeout < 0x100){
			timeout <<= 1;
		}
	}

	LOG_INFO, "Epoll reader thread stopped.";
}
void writerThreadProc(){
	LOG_INFO, "Epoll writer thread started.";

	std::signal(SIGPIPE, SIG_IGN);

	// 参考上面的注释。
	unsigned timeout = 1;
	while(atomicLoad(g_daemonRunning)){
		if(writerLoop(timeout)){
			timeout = 1;
		} else if(timeout < 0x100){
			timeout <<= 1;
		}
	}

	LOG_INFO, "Epoll writer thread stopped.";
}

}

void EpollDaemon::start(){
	if(atomicExchange(g_daemonRunning, true) != false){
		LOG_FATAL, "Only one daemon is allowed at the same time.";
		std::abort();
	}
	LOG_INFO, "Starting epoll daemon...";

	g_readerEpoll.reset(::epoll_create(4096));
	if(!g_readerEpoll){
		AUTO(desc, getErrorDesc());
		LOG_FATAL, "Error creating reader epoll: ", desc;
		std::abort();
	}
	g_writerEpoll.reset(::epoll_create(4096));
	if(!g_writerEpoll){
		AUTO(desc, getErrorDesc());
		LOG_FATAL, "Error creating writer epoll: ", desc;
		std::abort();
	}

	boost::thread(readerThreadProc).swap(g_readerThread);
	boost::thread(writerThreadProc).swap(g_writerThread);
}
void EpollDaemon::stop(){
	LOG_INFO, "Stopping epoll daemon...";

	atomicStore(g_daemonRunning, false);
	if(g_readerThread.joinable()){
		g_readerThread.join();
	}
	if(g_writerThread.joinable()){
		g_writerThread.join();
	}

	g_readable.clear();
	g_writeable.clear();
	g_servers.clear();
}

void EpollDaemon::notifyWriteable(boost::shared_ptr<TcpPeer> peer){
	assert(peer);

	addWritable(peer);
}
void EpollDaemon::addSocketServer(boost::shared_ptr<SocketServerBase> server){
	assert(server);

	const boost::mutex::scoped_lock lock(g_serverMutex);
	g_servers.insert(server);
}
void EpollDaemon::resetPeer(boost::shared_ptr<TcpPeer> peer){
	assert(peer);

	peer->forceShutdown();
	removeReadable(peer);
	removeWriteable(peer);
}
