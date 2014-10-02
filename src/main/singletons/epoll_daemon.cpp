#include "../../precompiled.hpp"
#include "epoll_daemon.hpp"
#include <boost/thread.hpp>
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
#include "../tcp_session_base.hpp"
#include "../socket_server.hpp"
#include "../multi_index_map.hpp"
#include "../profiler.hpp"
#include "config_file.hpp"
using namespace Poseidon;

namespace {

std::size_t g_tcpBufferSize		= 1024;
std::size_t g_epollBufferSize	= 256;
std::size_t g_maxEpollTimeout	= 100;

volatile bool g_running = false;
ScopedFile g_epoll;
boost::thread g_thread;

struct SessionMapElement {
	const boost::shared_ptr<TcpSessionBase> m_session;
	// 时间戳，零表示无数据可读/写。
	unsigned long long m_lastRead;
	unsigned long long m_lastWritten;

	SessionMapElement(boost::shared_ptr<TcpSessionBase> session,
		unsigned long long lastRead, unsigned long long lastWritten)
		: m_session(STD_MOVE(session)), m_lastRead(lastRead), m_lastWritten(lastWritten)
	{
	}
};

MULTI_INDEX_MAP(SessionMap, SessionMapElement,
	UNIQUE_MEMBER_INDEX(m_session),
	MULTI_MEMBER_INDEX(m_lastRead),
	MULTI_MEMBER_INDEX(m_lastWritten)
);

enum {
	IDX_SESSION,
	IDX_READ,
	IDX_WRITE,
};

boost::mutex g_sessionMutex;
SessionMap g_sessions;

boost::mutex g_serverMutex;
std::set<boost::shared_ptr<const SocketServerBase> > g_servers;

void addSession(const boost::shared_ptr<TcpSessionBase> &session){
	const AUTO(now, getMonoClock());
	std::pair<SessionMap::iterator, bool> result;
	{
		const boost::mutex::scoped_lock lock(g_sessionMutex);
		result = g_sessions.insert(SessionMapElement(session, now, now));
		if(!result.second){
			LOG_WARNING("Socket already in epoll?");
			return;
		}
	}
	::epoll_event event;
	event.events = EPOLLHUP | EPOLLERR | EPOLLRDHUP | EPOLLIN | EPOLLOUT | EPOLLET;
	event.data.ptr = session.get();
	if(::epoll_ctl(g_epoll.get(), EPOLL_CTL_ADD, session->getFd(), &event) != 0){
		const boost::mutex::scoped_lock lock(g_sessionMutex);
		g_sessions.erase(result.first);
		DEBUG_THROW(SystemError, errno);
	}
}
void touchSession(const boost::shared_ptr<TcpSessionBase> &session){
	const AUTO(now, getMonoClock());
	const boost::mutex::scoped_lock lock(g_sessionMutex);
	const AUTO(it, g_sessions.find<0>(session));
	if(it == g_sessions.end<0>()){
		LOG_WARNING("Socket not in epoll?");
		return;
	}
	g_sessions.setKey<IDX_SESSION, IDX_READ>(it, now);
	g_sessions.setKey<IDX_SESSION, IDX_WRITE>(it, now);
}
void removeSession(const boost::shared_ptr<TcpSessionBase> &session){
	if(::epoll_ctl(g_epoll.get(), EPOLL_CTL_DEL, session->getFd(), NULLPTR) != 0){
		LOG_WARNING("Error deleting from epoll. We can do nothing but ignore it.");
	}
	const boost::mutex::scoped_lock lock(g_sessionMutex);
	const AUTO(it, g_sessions.find<IDX_SESSION>(session));
	if(it == g_sessions.end<IDX_SESSION>()){
		LOG_WARNING("Socket not in epoll?");
		return;
	}
	g_sessions.erase<IDX_SESSION>(it);
}

void deepollReadable(const boost::shared_ptr<TcpSessionBase> &session){
	const AUTO(now, getMonoClock());
	const boost::mutex::scoped_lock lock(g_sessionMutex);
	const AUTO(it, g_sessions.find<IDX_SESSION>(session));
	if(it == g_sessions.end<IDX_SESSION>()){
		LOG_ERROR("Socket not in epoll?");
		return;
	}
	g_sessions.setKey<IDX_SESSION, IDX_READ>(it, now);
}
void reepollReadable(const boost::shared_ptr<TcpSessionBase> &session){
	const boost::mutex::scoped_lock lock(g_sessionMutex);
	const AUTO(it, g_sessions.find<IDX_SESSION>(session));
	if(it == g_sessions.end<IDX_SESSION>()){
		LOG_ERROR("Socket not in epoll?");
		return;
	}
	g_sessions.setKey<IDX_SESSION, IDX_READ>(it, 0);
}

void deepollWriteable(const boost::shared_ptr<TcpSessionBase> &session){
	const AUTO(now, getMonoClock());
	const boost::mutex::scoped_lock lock(g_sessionMutex);
	const AUTO(it, g_sessions.find<IDX_SESSION>(session));
	if(it == g_sessions.end<IDX_SESSION>()){
		LOG_ERROR("Socket not in epoll?");
		return;
	}
	g_sessions.setKey<IDX_SESSION, IDX_WRITE>(it, now);
}
void reepollWriteable(const boost::shared_ptr<TcpSessionBase> &session){
	const boost::mutex::scoped_lock lock(g_sessionMutex);
	const AUTO(it, g_sessions.find<IDX_SESSION>(session));
	if(it == g_sessions.end<IDX_SESSION>()){
		LOG_ERROR("Socket not in epoll?");
		return;
	}
	g_sessions.setKey<IDX_SESSION, IDX_WRITE>(it, 0);
}

void daemonLoop(){
	const boost::scoped_array<unsigned char>
		data(new unsigned char[g_tcpBufferSize]);
	const boost::scoped_array< ::epoll_event>
		events(new ::epoll_event[g_epollBufferSize]);
	std::size_t epollTimeout = 0;

	std::vector<boost::shared_ptr<TcpSessionBase> > sessions;
	std::vector<boost::shared_ptr<const SocketServerBase> > servers;

	while(atomicLoad(g_running)){
		// 第一部分，处理可接收的数据。
		{
			sessions.clear();
			const boost::mutex::scoped_lock lock(g_sessionMutex);
			for(AUTO(it, g_sessions.upperBound<IDX_READ>(0));
				it != g_sessions.end<IDX_READ>(); ++it)
			{
				sessions.push_back(it->m_session);
			}
		}
		for(AUTO(it, sessions.begin()); it != sessions.end(); ++it){
			epollTimeout = 0;
			const AUTO_REF(session, *it);
			try {
				const ::ssize_t bytesRead = ::recv(session->getFd(),
					data.get(), g_tcpBufferSize, MSG_NOSIGNAL);
				if(bytesRead < 0){
					if(errno == EINTR){
						continue;
					}
					if(errno == EAGAIN){
						reepollReadable(session);
						continue;
					}
					DEBUG_THROW(SystemError, errno);
				} else if(bytesRead == 0){
					reepollReadable(session);
					continue;
				}
				if(session->hasBeenShutdown()){
					continue;
				}
				LOG_DEBUG("Read ", bytesRead, " byte(s) from ", session->getRemoteIp());
				session->onReadAvail(data.get(), bytesRead);
			} catch(std::exception &e){
				LOG_ERROR("std::exception thrown while dispatching data: what = ", e.what());
				session->shutdown();
			} catch(...){
				LOG_ERROR("Unknown exception thrown while dispatching data.");
				session->shutdown();
			}
		}

		// 第二部分，处理可发送的数据。
		{
			sessions.clear();
			const boost::mutex::scoped_lock lock(g_sessionMutex);
			for(AUTO(it, g_sessions.upperBound<IDX_WRITE>(0));
				it != g_sessions.end<IDX_WRITE>(); ++it)
			{
				sessions.push_back(it->m_session);
			}
		}
		for(AUTO(it, sessions.begin()); it != sessions.end(); ++it){
			epollTimeout = 0;
			const AUTO_REF(session, *it);
			try {
				std::size_t bytesToWrite;
				bool shutdown;
				{
					boost::mutex::scoped_lock sessionLock;
					bytesToWrite = session->peekWriteAvail(sessionLock,
						data.get(), g_tcpBufferSize);
					shutdown = session->hasBeenShutdown();
					if(bytesToWrite == 0){
						if(!shutdown){
							reepollWriteable(session);
						}
					}
				}
				if(bytesToWrite == 0){
					if(shutdown){
						removeSession(session);
					}
					continue;
				}
				const ::ssize_t bytesWritten = ::send(session->getFd(),
					data.get(), bytesToWrite, MSG_NOSIGNAL);
				if(bytesWritten < 0){
					if(errno == EINTR){
						continue;
					}
					if(errno == EAGAIN){
						reepollWriteable(session);
						continue;
					}
					DEBUG_THROW(SystemError, errno);
				} else if(bytesWritten == 0){
					reepollWriteable(session);
					continue;
				}
				LOG_DEBUG("Wrote ", bytesWritten, " byte(s) to ", session->getRemoteIp());
				session->notifyWritten(bytesWritten);
			} catch(std::exception &e){
				LOG_ERROR("std::exception thrown while writing socket: what = ", e.what());
				removeSession(session);
			} catch(...){
				LOG_ERROR("Unknown exception thrown while writing socket.");
				removeSession(session);
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
				AUTO(session, (*it)->tryAccept());
				if(!session){
					continue;
				}
				LOG_DEBUG("Accepted socket connection from ", session->getRemoteIp());
				epollTimeout = 0;
				addSession(session);
			} catch(std::exception &e){
				LOG_ERROR("std::exception thrown while accepting client: what = ", e.what());
			} catch(...){
				LOG_ERROR("Unknown exception thrown while accepting client.");
			}
		}

		// 第四部分，检测新的数据。
		const int ready = ::epoll_wait(g_epoll.get(), events.get(), g_epollBufferSize, epollTimeout);
		if(ready < 0){
			const AUTO(desc, getErrorDesc());
			LOG_ERROR("::epoll_wait() failed: ", desc);
		} else for(unsigned i = 0; i < (unsigned)ready; ++i){
			::epoll_event &event = events[i];
			const AUTO(session, static_cast<TcpSessionBase *>(event.data.ptr)->
				virtualSharedFromThis<TcpSessionBase>());
			try {
				if(event.events & EPOLLHUP){
					LOG_INFO("Socket has been hung up. Remove it.");
					removeSession(session);
					continue;
				}
				if(event.events & EPOLLERR){
					int err;
					::socklen_t errLen = sizeof(err);
					if(::getsockopt(session->getFd(), SOL_SOCKET, SO_ERROR, &err, &errLen) != 0){
						err = errno;
					}
					const AUTO(desc, getErrorDesc());
					LOG_WARNING("Socket error: ", desc);
					removeSession(session);
					continue;
				}

				if(event.events & EPOLLRDHUP){
					session->shutdown();
					event.events |= EPOLLIN;
					event.events |= EPOLLOUT;
				}
				if(event.events & EPOLLIN){
					deepollReadable(session);
				}
				if(event.events & EPOLLOUT){
					deepollWriteable(session);
				}
			} catch(std::exception &e){
				LOG_ERROR("std::exception thrown while epolling: what = ", e.what());
				removeSession(session);
			} catch(...){
				LOG_ERROR("Unknown exception thrown while epolling.");
				removeSession(session);
			}
		}

		// 二次指数回退算法。如果有连接接入（忙），epoll 等待时间就短一些；反之（闲）亦然。
		if(epollTimeout == 0){
			epollTimeout = 1;
		} else {
			epollTimeout <<= 1;
		}
		if(epollTimeout > g_maxEpollTimeout){
			epollTimeout = g_maxEpollTimeout;
		}
	}
}

void threadProc(){
	PROFILE_ME;
	Log::setThreadTag(Log::TAG_EPOLL);
	LOG_INFO("Epoll daemon started.");

	g_tcpBufferSize =
		ConfigFile::get<std::size_t>("tcp_buffer_size", g_tcpBufferSize);
	LOG_DEBUG("TCP buffer size = ", g_tcpBufferSize);

	g_epollBufferSize =
		ConfigFile::get<std::size_t>("epoll_buffer_size", g_epollBufferSize);
	LOG_DEBUG("Epoll buffer size = ", g_epollBufferSize);

	g_maxEpollTimeout =
		ConfigFile::get<std::size_t>("max_epoll_timeout", g_maxEpollTimeout);
	LOG_DEBUG("Max epoll timeout = ", g_maxEpollTimeout);

	daemonLoop();

	LOG_INFO("Epoll daemon stopped.");
}

}

void EpollDaemon::start(){
	if(atomicExchange(g_running, true) != false){
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

	boost::thread(threadProc).swap(g_thread);
}
void EpollDaemon::stop(){
	LOG_INFO("Stopping epoll daemon...");

	atomicStore(g_running, false);
	if(g_thread.joinable()){
		g_thread.join();
	}

	g_sessions.clear();
	g_servers.clear();
}

void EpollDaemon::addSession(const boost::shared_ptr<TcpSessionBase> &session){
	assert(session);

	if(session->hasBeenShutdown()){
		return;
	}
	addSession(session);
}
void EpollDaemon::refreshSession(const boost::shared_ptr<TcpSessionBase> &session){
	assert(session);

	if(session->hasBeenShutdown()){
		return;
	}
	touchSession(session);
}
void EpollDaemon::addSocketServer(boost::shared_ptr<SocketServerBase> server){
	assert(server);

	{
		const boost::mutex::scoped_lock lock(g_serverMutex);
		g_servers.insert(server);
	}
}
