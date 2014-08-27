#include "../../precompiled.hpp"
#include "epoll_daemon.hpp"
#include <boost/thread.hpp>
#include <boost/make_shared.hpp>
#include <boost/noncopyable.hpp>
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
#include "../socket_server.hpp"
#include "../multi_index_map.hpp"
using namespace Poseidon;

namespace {

volatile bool g_daemonRunning = false;
ScopedFile g_epoll;
boost::thread g_daemonThread;

class EpollRaii : boost::noncopyable {
private:
	TcpPeer *const m_peer;

public:
	explicit EpollRaii(TcpPeer *peer)
		: m_peer(peer)
	{
		::epoll_event event;
		event.events = EPOLLHUP | EPOLLERR | EPOLLIN | EPOLLOUT | EPOLLET;
		event.data.ptr = m_peer;
		if(::epoll_ctl(g_epoll.get(), EPOLL_CTL_ADD, m_peer->getFd(), &event) != 0){
			DEBUG_THROW(SystemError, errno);
		}
	}
	~EpollRaii(){
		if(::epoll_ctl(g_epoll.get(), EPOLL_CTL_DEL, m_peer->getFd(), NULL) != 0){
			LOG_WARNING("Deleting from epoll failed. We can do nothing but ignore it.");
		}
	}
};

struct PeerMapElement {
	const boost::shared_ptr<TcpPeer> m_peer;
	// 时间戳，零表示无数据可读/写。
	unsigned long long m_lastRead;
	unsigned long long m_lastWritten;

	const boost::shared_ptr<EpollRaii> m_epollRaii;

	explicit PeerMapElement(boost::shared_ptr<TcpPeer> peer,
		unsigned long long lastRead = 0, unsigned long long lastWritten = 0)
		: m_peer(peer), m_lastRead(lastRead), m_lastWritten(lastWritten)
		, m_epollRaii(boost::make_shared<EpollRaii>(peer.get()))
	{
	}
};

MULTI_INDEX_MAP(PeerMap, PeerMapElement,
	UNIQUE_INDEX(m_peer),
	MULTI_INDEX(m_lastRead),
	MULTI_INDEX(m_lastWritten)
);

enum {
	IDX_PEER,
	IDX_READ,
	IDX_WRITE,
};

boost::mutex g_peerMutex;
PeerMap g_peers;

boost::mutex g_serverMutex;
std::set<boost::shared_ptr<const SocketServerBase> > g_servers;

void addPeer(const boost::shared_ptr<TcpPeer> &peer){
	const boost::mutex::scoped_lock lock(g_peerMutex);
	g_peers.insert(PeerMapElement(peer));
}
void removePeer(const boost::shared_ptr<TcpPeer> &peer){
	const boost::mutex::scoped_lock lock(g_peerMutex);
	const AUTO(it, g_peers.find<IDX_PEER>(peer));
	if(it == g_peers.end<IDX_PEER>()){
		LOG_ERROR("Socket not in epoll?");
		return;
	}
	g_peers.erase<IDX_PEER>(it);
}

void deepollReadable(const boost::shared_ptr<TcpPeer> &peer){
	const unsigned long long now = getMonoClock();
	const boost::mutex::scoped_lock lock(g_peerMutex);
	const AUTO(it, g_peers.find<IDX_PEER>(peer));
	if(it == g_peers.end<IDX_PEER>()){
		LOG_ERROR("Socket not in epoll?");
		return;
	}
	g_peers.setKey<IDX_PEER, IDX_READ>(it, now);
}
void reepollReadable(const boost::shared_ptr<TcpPeer> &peer){
	const boost::mutex::scoped_lock lock(g_peerMutex);
	const AUTO(it, g_peers.find<IDX_PEER>(peer));
	if(it == g_peers.end<IDX_PEER>()){
		LOG_ERROR("Socket not in epoll?");
		return;
	}
	g_peers.setKey<IDX_PEER, IDX_READ>(it, 0);
}

void deepollWriteable(const boost::shared_ptr<TcpPeer> &peer){
	const unsigned long long now = getMonoClock();
	const boost::mutex::scoped_lock lock(g_peerMutex);
	const AUTO(it, g_peers.find<IDX_PEER>(peer));
	if(it == g_peers.end<IDX_PEER>()){
		LOG_ERROR("Socket not in epoll?");
		return;
	}
	g_peers.setKey<IDX_PEER, IDX_WRITE>(it, now);
}
void reepollWriteable(const boost::shared_ptr<TcpPeer> &peer){
	const boost::mutex::scoped_lock lock(g_peerMutex);
	const AUTO(it, g_peers.find<IDX_PEER>(peer));
	if(it == g_peers.end<IDX_PEER>()){
		LOG_ERROR("Socket not in epoll?");
		return;
	}
	g_peers.setKey<IDX_PEER, IDX_WRITE>(it, 0);
}

void threadProc(){
	LOG_INFO("Epoll daemon thread started.");

	std::vector<boost::shared_ptr<TcpPeer> > peers;
	std::vector<boost::shared_ptr<const SocketServerBase> > servers;
	unsigned char data[1024];

	while(atomicLoad(g_daemonRunning)){
		// 第一部分，处理可接收的数据。
		{
			peers.clear();
			const boost::mutex::scoped_lock lock(g_peerMutex);
			for(AUTO(it, g_peers.upperBound<IDX_READ>(0)); it != g_peers.end<IDX_READ>(); ++it){
				peers.push_back(it->m_peer);
			}
		}
		for(AUTO(it, peers.begin()); it != peers.end(); ++it){
			const AUTO_REF(peer, *it);
			try {
				const ::ssize_t bytesRead = ::recv(peer->getFd(), data, sizeof(data), MSG_NOSIGNAL);
				if(bytesRead < 0){
					if(errno == EINTR){
						continue;
					}
					if(errno == EAGAIN){
						reepollReadable(peer);
						continue;
					}
					DEBUG_THROW(SystemError, errno);
				} else if(bytesRead == 0){
					LOG_INFO("Socket has been closed by peer. Remove it.");
					peer->onRemoteClose();
					removePeer(peer);
					continue;
				} else {
					peer->onReadAvail(data, bytesRead);
				}
			} catch(Exception &e){
				LOG_ERROR("Exception thrown while dispatching data: file = ", e.file(),
					", line = ", e.line(), ", what = ", e.what());
				removePeer(peer);
			} catch(std::exception &e){
				LOG_ERROR("std::exception thrown while dispatching data: what = ", e.what());
				removePeer(peer);
			} catch(...){
				LOG_ERROR("Unknown exception thrown while dispatching data.");
				removePeer(peer);
			}
		}

		// 第二部分，处理可发送的数据。
		{
			peers.clear();
			const boost::mutex::scoped_lock lock(g_peerMutex);
			for(AUTO(it, g_peers.upperBound<IDX_WRITE>(0)); it != g_peers.end<IDX_WRITE>(); ++it){
				peers.push_back(it->m_peer);
			}
		}
		for(AUTO(it, peers.begin()); it != peers.end(); ++it){
			const AUTO_REF(peer, *it);
			try {
				std::size_t bytesToWrite;
				{
					boost::mutex::scoped_lock peerLock;
					bytesToWrite = peer->peekWriteAvail(peerLock, data, sizeof(data));
					if(bytesToWrite == 0){
						reepollWriteable(peer);
						continue;
					}
				}
				const ::ssize_t bytesWritten = ::send(peer->getFd(), data, bytesToWrite, MSG_NOSIGNAL);
				if(bytesWritten < 0){
					if(errno == EINTR){
						continue;
					}
					if(errno == EAGAIN){
						reepollWriteable(peer);
						continue;
					}
					DEBUG_THROW(SystemError, errno);
				} else if(bytesWritten == 0){
					reepollWriteable(peer);
					continue;
				}
				peer->notifyWritten(bytesWritten);
			} catch(Exception &e){
				LOG_ERROR("Exception thrown while writing socket: file = ", e.file(),
					", line = ", e.line(), ", what = ", e.what());
				removePeer(peer);
			} catch(std::exception &e){
				LOG_ERROR("std::exception thrown while writing socket: what = ", e.what());
				removePeer(peer);
			} catch(...){
				LOG_ERROR("Unknown exception thrown while writing socket.");
				removePeer(peer);
			}
		}

		// 第三部分，侦听新的连接。
		{
			servers.clear();
			const boost::mutex::scoped_lock lock(g_serverMutex);
			std::copy(g_servers.begin(), g_servers.end(), std::back_inserter(servers));
		}
		for(AUTO(it, servers.begin()); it != servers.end(); ++it){
			try {
				AUTO(peer, (*it)->tryAccept());
				if(!peer){
					continue;
				}
				addPeer(peer);
			} catch(Exception &e){
				LOG_ERROR("Exception thrown while accepting client: file = ", e.file(),
					", line = ", e.line(), ", what = ", e.what());
			} catch(std::exception &e){
				LOG_ERROR("std::exception thrown while accepting client: what = ", e.what());
			} catch(...){
				LOG_ERROR("Unknown exception thrown while accepting client.");
			}
		}

		// 第四部分，检测新的数据。
		::epoll_event events[256];
		const int ready = ::epoll_wait(g_epoll.get(), events, COUNT_OF(events), 100);
		if(ready < 0){
			const AUTO(desc, getErrorDesc());
			LOG_ERROR("::epoll_wait() failed: ", desc);
		} else for(unsigned i = 0; i < (unsigned)ready; ++i){
			::epoll_event &event = events[i];
			const AUTO(peer, static_cast<TcpPeer *>(event.data.ptr)->virtualSharedFromThis<TcpPeer>());
			try {
				if(event.events & EPOLLHUP){
					LOG_INFO("Socket has been hung up. Remove it.");
					removePeer(peer);
					continue;
				}
				if(event.events & EPOLLERR){
					int err;
					::socklen_t errLen = sizeof(err);
					if(::getsockopt(peer->getFd(), SOL_SOCKET, SO_ERROR, &err, &errLen) != 0){
						err = errno;
					}
					const AUTO(desc, getErrorDesc());
					LOG_WARNING("Socket error: ", desc);
					removePeer(peer);
					continue;
				}
				if(event.events & EPOLLIN){
					deepollReadable(peer);
				}
				if(event.events & EPOLLOUT){
					deepollWriteable(peer);
				}
			} catch(Exception &e){
				LOG_ERROR("Exception thrown while epolling: file = ", e.file(),
					", line = ", e.line(), ", what = ", e.what());
				removePeer(peer);
			} catch(std::exception &e){
				LOG_ERROR("std::exception thrown while epolling: what = ", e.what());
				removePeer(peer);
			} catch(...){
				LOG_ERROR("Unknown exception thrown while epolling.");
				removePeer(peer);
			}
		}
	}

	LOG_INFO("Epoll daemon thread stopped.");
}

}

void EpollDaemon::start(){
	if(atomicExchange(g_daemonRunning, true) != false){
		LOG_FATAL("Only one daemon is allowed at the same time.");
		std::abort();
	}
	LOG_INFO("Starting epoll daemon...");

	g_epoll.reset(::epoll_create(4096));
	if(!g_epoll){
		AUTO(desc, getErrorDesc());
		LOG_FATAL("Error creating epoll: ", desc);
		std::abort();
	}

	boost::thread(threadProc).swap(g_daemonThread);
}
void EpollDaemon::stop(){
	LOG_INFO("Stopping epoll daemon...");

	atomicStore(g_daemonRunning, false);
	if(g_daemonThread.joinable()){
		g_daemonThread.join();
	}

	g_peers.clear();
	g_servers.clear();
}

void EpollDaemon::refreshPeer(boost::shared_ptr<TcpPeer> peer){
	assert(peer);

	if(peer->hasBeenShutdown()){
		return;
	}
	const unsigned long long now = getMonoClock();
	const boost::mutex::scoped_lock lock(g_peerMutex);
	const AUTO(it, g_peers.find<0>(peer));
	if(it == g_peers.end<0>()){
		g_peers.insert(PeerMapElement(peer, now, now));
	} else {
		g_peers.setKey<IDX_PEER, IDX_READ>(it, now);
		g_peers.setKey<IDX_PEER, IDX_WRITE>(it, now);
	}
}
void EpollDaemon::addSocketServer(boost::shared_ptr<SocketServerBase> server){
	assert(server);

	const boost::mutex::scoped_lock lock(g_serverMutex);
	g_servers.insert(server);
}
