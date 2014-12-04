// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014, LH_Mouse. All wrongs reserved.

#include "../precompiled.hpp"
#include "epoll_daemon.hpp"
#include "main_config.hpp"
#include <boost/thread.hpp>
#include <sys/epoll.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <errno.h>
#include <string.h>
#include "../log.hpp"
#include "../atomic.hpp"
#include "../utilities.hpp"
#include "../exception.hpp"
#include "../tcp_session_base.hpp"
#include "../socket_server_base.hpp"
#include "../multi_index_map.hpp"
#include "../profiler.hpp"
using namespace Poseidon;

namespace {

std::size_t	g_dataBufferSize	= 1024;
std::size_t	g_eventBufferSize	= 256;
std::size_t	g_maxTimeout		= 100;

volatile bool g_running = false;
UniqueFile g_epoll;
boost::thread g_thread;

struct SessionMapElement {
	const boost::shared_ptr<TcpSessionBase> session;
	// 时间戳，零表示无数据可读/写。
	unsigned long long lastRead;
	unsigned long long lastWritten;

	SessionMapElement(boost::shared_ptr<TcpSessionBase> session_,
		unsigned long long lastRead_, unsigned long long lastWritten_)
		: session(STD_MOVE(session_)), lastRead(lastRead_), lastWritten(lastWritten_)
	{
	}

#ifndef POSEIDON_CXX11
	// C++03 不提供转移构造函数，但是我们在这里不使用它，不需要定义。
	SessionMapElement(Move<SessionMapElement> rhs);
#endif
};

MULTI_INDEX_MAP(SessionMap, SessionMapElement,
	UNIQUE_MEMBER_INDEX(session)
	MULTI_MEMBER_INDEX(lastRead)
	MULTI_MEMBER_INDEX(lastWritten)
);

enum {
	IDX_SESSION,
	IDX_READ,
	IDX_WRITE,
};

struct HexEncoder {
	const void *const read;
	const std::size_t size;

	HexEncoder(const void *read_, std::size_t size_)
		: read(read_), size(size_)
	{
	}
};

std::ostream &operator<<(std::ostream &os, const HexEncoder &rhs){
	const AUTO(data, reinterpret_cast<const unsigned char *>(rhs.read));
	for(std::size_t i = 0; i < rhs.size; ++i){
		char temp[16];
		unsigned len = std::sprintf(temp, "%02X ", data[i]);
		os.write(temp, len);
	}
	return os;
}

boost::mutex g_sessionMutex;
SessionMap g_sessions;

boost::mutex g_serverMutex;
std::list<boost::weak_ptr<const SocketServerBase> > g_servers;

void add(const boost::shared_ptr<TcpSessionBase> &session){
	const AUTO(now, getMonoClock());
	std::pair<SessionMap::iterator, bool> result;
	{
		const boost::mutex::scoped_lock lock(g_sessionMutex);
		result = g_sessions.insert(SessionMapElement(session, now, now));
		if(!result.second){
			LOG_POSEIDON_WARN("Socket already in epoll?");
			return;
		}
	}
	::epoll_event event;
	event.events = EPOLLIN | EPOLLOUT | EPOLLET;
	event.data.ptr = session.get();
	if(::epoll_ctl(g_epoll.get(), EPOLL_CTL_ADD, session->getFd(), &event) != 0){
		const boost::mutex::scoped_lock lock(g_sessionMutex);
		g_sessions.erase(result.first);
		DEBUG_THROW(SystemError);
	}
}
void touch(const boost::shared_ptr<TcpSessionBase> &session){
	const AUTO(now, getMonoClock());
	const boost::mutex::scoped_lock lock(g_sessionMutex);
	const AUTO(it, g_sessions.find<0>(session));
	if(it == g_sessions.end<0>()){
		LOG_POSEIDON_WARN("Socket not in epoll?");
		return;
	}
	g_sessions.setKey<IDX_SESSION, IDX_READ>(it, now);
	g_sessions.setKey<IDX_SESSION, IDX_WRITE>(it, now);
}
void remove(const boost::shared_ptr<TcpSessionBase> &session){
	if(::epoll_ctl(g_epoll.get(), EPOLL_CTL_DEL, session->getFd(), NULLPTR) != 0){
		LOG_POSEIDON_WARN("Error deleting from epoll. We can do nothing but ignore it.");
	}
	const boost::mutex::scoped_lock lock(g_sessionMutex);
	const AUTO(it, g_sessions.find<IDX_SESSION>(session));
	if(it == g_sessions.end<IDX_SESSION>()){
		LOG_POSEIDON_WARN("Socket not in epoll?");
		return;
	}
	g_sessions.erase<IDX_SESSION>(it);
}

void deepollReadable(const boost::shared_ptr<TcpSessionBase> &session){
	const AUTO(now, getMonoClock());
	const boost::mutex::scoped_lock lock(g_sessionMutex);
	const AUTO(it, g_sessions.find<IDX_SESSION>(session));
	if(it == g_sessions.end<IDX_SESSION>()){
		LOG_POSEIDON_ERROR("Socket not in epoll?");
		return;
	}
	g_sessions.setKey<IDX_SESSION, IDX_READ>(it, now);
}
void reepollReadable(const boost::shared_ptr<TcpSessionBase> &session){
	const boost::mutex::scoped_lock lock(g_sessionMutex);
	const AUTO(it, g_sessions.find<IDX_SESSION>(session));
	if(it == g_sessions.end<IDX_SESSION>()){
		LOG_POSEIDON_ERROR("Socket not in epoll?");
		return;
	}
	g_sessions.setKey<IDX_SESSION, IDX_READ>(it, 0);
}

void deepollWriteable(const boost::shared_ptr<TcpSessionBase> &session){
	const AUTO(now, getMonoClock());
	const boost::mutex::scoped_lock lock(g_sessionMutex);
	const AUTO(it, g_sessions.find<IDX_SESSION>(session));
	if(it == g_sessions.end<IDX_SESSION>()){
		LOG_POSEIDON_ERROR("Socket not in epoll?");
		return;
	}
	g_sessions.setKey<IDX_SESSION, IDX_WRITE>(it, now);
}
void reepollWriteable(const boost::shared_ptr<TcpSessionBase> &session){
	const boost::mutex::scoped_lock lock(g_sessionMutex);
	const AUTO(it, g_sessions.find<IDX_SESSION>(session));
	if(it == g_sessions.end<IDX_SESSION>()){
		LOG_POSEIDON_ERROR("Socket not in epoll?");
		return;
	}
	g_sessions.setKey<IDX_SESSION, IDX_WRITE>(it, 0);
}

void daemonLoop(){
	const boost::scoped_array<unsigned char> data(new unsigned char[g_dataBufferSize]);
	const boost::scoped_array< ::epoll_event> events(new ::epoll_event[g_eventBufferSize]);

	std::size_t epollTimeout = 0;
	std::vector<boost::shared_ptr<TcpSessionBase> > sessions;
	std::vector<boost::shared_ptr<const SocketServerBase> > servers;

	while(atomicLoad(g_running)){
		// 第一部分，处理可接收的数据。
		{
			const boost::mutex::scoped_lock lock(g_sessionMutex);
			for(AUTO(it, g_sessions.upperBound<IDX_READ>(0)); it != g_sessions.end<IDX_READ>(); ++it){
				sessions.push_back(it->session);
			}
		}
		for(AUTO(it, sessions.begin()); it != sessions.end(); ++it){
			const AUTO_REF(session, *it);
			try {
				if(session->hasBeenShutdown()){
					continue;
				}
				const ::ssize_t bytesRead = session->syncRead(data.get(), g_dataBufferSize);
				if(bytesRead < 0){
					if(errno == EINTR){
						continue;
					}
					if(errno == EAGAIN){
						reepollReadable(session);
						continue;
					}
					DEBUG_THROW(SystemError);
				} else if(bytesRead == 0){
					LOG_POSEIDON_INFO("Connection closed: remote = ", session->getRemoteInfo());
					session->send(StreamBuffer(), true);
					continue;
				}
				LOG_POSEIDON_TRACE("Read ", bytesRead, " byte(s) from ", session->getRemoteInfo(),
					", hex = ", HexEncoder(data.get(), bytesRead));
			} catch(std::exception &e){
				LOG_POSEIDON_ERROR("std::exception thrown while dispatching data: what = ", e.what());
				session->send(StreamBuffer(), true);
			} catch(...){
				LOG_POSEIDON_ERROR("Unknown exception thrown while dispatching data.");
				session->send(StreamBuffer(), true);
			}
			epollTimeout = 0;
		}
		sessions.clear();

		// 第二部分，处理可发送的数据。
		{
			const boost::mutex::scoped_lock lock(g_sessionMutex);
			for(AUTO(it, g_sessions.upperBound<IDX_WRITE>(0)); it != g_sessions.end<IDX_WRITE>(); ++it){
				sessions.push_back(it->session);
			}
		}
		for(AUTO(it, sessions.begin()); it != sessions.end(); ++it){
			const AUTO_REF(session, *it);
			try {
				::ssize_t bytesWritten;
				bool shutdown;
				{
					boost::mutex::scoped_lock sessionLock;
					bytesWritten = session->syncWrite(sessionLock, data.get(), g_dataBufferSize);
					shutdown = session->hasBeenShutdown();
					if(bytesWritten == 0){
						if(!shutdown){
							reepollWriteable(session);
						}
					}
				}
				if(bytesWritten < 0){
					if(errno == EINTR){
						continue;
					}
					if(errno == EAGAIN){
						reepollWriteable(session);
						continue;
					}
					DEBUG_THROW(SystemError);
				}
				if(bytesWritten == 0){
					if(shutdown){
						remove(session);
					}
					continue;
				}
				LOG_POSEIDON_TRACE("Wrote ", bytesWritten, " byte(s) to ", session->getRemoteInfo(),
					", hex = ", HexEncoder(data.get(), bytesWritten));
			} catch(std::exception &e){
				LOG_POSEIDON_ERROR("std::exception thrown while writing socket: what = ", e.what());
				remove(session);
			} catch(...){
				LOG_POSEIDON_ERROR("Unknown exception thrown while writing socket.");
				remove(session);
			}
			epollTimeout = 0;
		}
		sessions.clear();

		// 第三部分，侦听新的连接。
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
		for(AUTO(it, servers.begin()); it != servers.end(); ++it){
			try {
				if(!(*it)->poll()){
					continue;
				}
			} catch(std::exception &e){
				LOG_POSEIDON_ERROR("std::exception thrown while accepting connection: what = ", e.what());
			} catch(...){
				LOG_POSEIDON_ERROR("Unknown exception thrown while accepting connection.");
			}
			epollTimeout = 0;
		}
		servers.clear();

		// 第四部分，检测新的数据。
		const int ready = ::epoll_wait(g_epoll.get(), events.get(), g_eventBufferSize, epollTimeout);
		if(ready < 0){
			const AUTO(desc, getErrorDesc());
			LOG_POSEIDON_ERROR("::epoll_wait() failed: ", desc);
		} else for(unsigned i = 0; i < (unsigned)ready; ++i){
			::epoll_event &event = events[i];
			const AUTO(session,
				static_cast<TcpSessionBase *>(event.data.ptr)->virtualSharedFromThis<TcpSessionBase>());
			try {
				if(event.events & EPOLLHUP){
					LOG_POSEIDON_INFO("Socket hung up, remote is ", session->getRemoteInfo());
					remove(session);
					continue;
				}
				if(event.events & EPOLLERR){
					int err;
					::socklen_t errLen = sizeof(err);
					if(::getsockopt(session->getFd(), SOL_SOCKET, SO_ERROR, &err, &errLen) != 0){
						err = errno;
					}
					const AUTO(desc, getErrorDesc());
					LOG_POSEIDON_WARN("Socket error: ", desc);
					remove(session);
					continue;
				}

				if(event.events & EPOLLIN){
					deepollReadable(session);
				}
				if(event.events & EPOLLOUT){
					deepollWriteable(session);
				}
			} catch(std::exception &e){
				LOG_POSEIDON_ERROR("std::exception thrown while epolling: what = ", e.what());
				remove(session);
			} catch(...){
				LOG_POSEIDON_ERROR("Unknown exception thrown while epolling.");
				remove(session);
			}
		}

		// 二次指数回退算法。如果有连接接入（忙），epoll 等待时间就短一些；反之（闲）亦然。
		if(epollTimeout == 0){
			epollTimeout = 1;
		} else {
			epollTimeout <<= 1;
		}
		if(epollTimeout > g_maxTimeout){
			epollTimeout = g_maxTimeout;
		}
	}

	SessionMap remaining;
	{
		const boost::mutex::scoped_lock lock(g_sessionMutex);
		remaining.swap(g_sessions);
	}
	if(!remaining.empty()){
		LOG_POSEIDON_DEBUG("Flushing data on ", remaining.size(), " socket(s).");

		for(AUTO(it, remaining.begin()); it != remaining.end(); ++it){
			const AUTO_REF(session, it->session);
			try {
				::ssize_t bytesWritten;
				for(;;){
					{
						boost::mutex::scoped_lock sessionLock;
						bytesWritten = session->syncWrite(sessionLock, data.get(), g_dataBufferSize);
					}
					if(bytesWritten <= 0){
						break;
					}
					LOG_POSEIDON_TRACE("Flushed ", bytesWritten, " byte(s) to ", session->getRemoteInfo(),
						", hex = ", HexEncoder(data.get(), bytesWritten));
				}
			} catch(std::exception &e){
				LOG_POSEIDON_ERROR("std::exception thrown while flush data: what = ", e.what());
			} catch(...){
				LOG_POSEIDON_ERROR("Unknown exception thrown while flush data.");
			}
		}
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
	if(atomicExchange(g_running, true) != false){
		LOG_POSEIDON_FATAL("Only one daemon is allowed at the same time.");
		std::abort();
	}
	LOG_POSEIDON_INFO("Starting epoll daemon...");

	AUTO_REF(conf, MainConfig::getConfigFile());

	conf.get(g_dataBufferSize, "epoll_data_buffer_size");
	LOG_POSEIDON_DEBUG("Data buffer size = ", g_dataBufferSize);

	conf.get(g_eventBufferSize, "epoll_event_buffer_size");
	LOG_POSEIDON_DEBUG("Event buffer size = ", g_eventBufferSize);

	conf.get(g_maxTimeout, "epoll_max_timeout");
	LOG_POSEIDON_DEBUG("Max timeout = ", g_maxTimeout);

	if(!g_epoll.reset(::epoll_create(4096))){
		AUTO(desc, getErrorDesc());
		LOG_POSEIDON_FATAL("Error creating epoll: ", desc);
		std::abort();
	}

	boost::thread(threadProc).swap(g_thread);
}
void EpollDaemon::stop(){
	if(atomicExchange(g_running, false) == false){
		return;
	}
	LOG_POSEIDON_INFO("Stopping epoll daemon...");

	if(g_thread.joinable()){
		g_thread.join();
	}
	g_sessions.clear();
	g_servers.clear();
}

std::vector<EpollSnapshotItem> EpollDaemon::snapshot(){
	std::vector<EpollSnapshotItem> ret;
	const AUTO(now, getMonoClock());
	const boost::mutex::scoped_lock lock(g_sessionMutex);
	ret.reserve(g_sessions.size());
	for(AUTO(it, g_sessions.begin()); it != g_sessions.end(); ++it){
		ret.push_back(EpollSnapshotItem());
		AUTO_REF(item, ret.back());
		item.remote = it->session->getRemoteInfo();
		item.local = it->session->getLocalInfo();
		item.usOnline = now - it->session->getCreatedTime();
	}
	return ret;
}

void EpollDaemon::addSession(const boost::shared_ptr<TcpSessionBase> &session){
	if(session->hasBeenShutdown()){
		return;
	}
	add(session);
}
void EpollDaemon::touchSession(const boost::shared_ptr<TcpSessionBase> &session){
	if(session->hasBeenShutdown()){
		return;
	}
	touch(session);
}

void EpollDaemon::registerServer(boost::shared_ptr<SocketServerBase> server){
	const boost::mutex::scoped_lock lock(g_serverMutex);
	g_servers.push_back(STD_MOVE(server));
}
