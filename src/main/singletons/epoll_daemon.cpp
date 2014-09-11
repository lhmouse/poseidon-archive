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
#include "config_file.hpp"
using namespace Poseidon;

namespace {

volatile bool g_daemonRunning = false;
ScopedFile g_epoll;
boost::thread g_daemonThread;

class EpollRaii : boost::noncopyable {
private:
	TcpSessionBase *const m_session;

public:
	explicit EpollRaii(TcpSessionBase *session)
		: m_session(session)
	{
		::epoll_event event;
		event.events = EPOLLHUP | EPOLLERR | EPOLLRDHUP | EPOLLIN | EPOLLOUT | EPOLLET;
		event.data.ptr = m_session;
		if(::epoll_ctl(g_epoll.get(), EPOLL_CTL_ADD, m_session->getFd(), &event) != 0){
			DEBUG_THROW(SystemError, errno);
		}
	}
	~EpollRaii(){
		if(::epoll_ctl(g_epoll.get(), EPOLL_CTL_DEL, m_session->getFd(), NULLPTR) != 0){
			LOG_WARNING("Deleting from epoll failed. We can do nothing but ignore it.");
		}
	}
};

struct SessionMapElement {
	const boost::shared_ptr<TcpSessionBase> m_session;
	// 时间戳，零表示无数据可读/写。
	unsigned long long m_lastRead;
	unsigned long long m_lastWritten;

	const boost::shared_ptr<EpollRaii> m_epollRaii;

	explicit SessionMapElement(boost::shared_ptr<TcpSessionBase> session,
		unsigned long long lastRead = 0, unsigned long long lastWritten = 0)
		: m_session(STD_MOVE(session)), m_lastRead(lastRead), m_lastWritten(lastWritten)
		, m_epollRaii(boost::make_shared<EpollRaii>(session.get()))
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
	const boost::mutex::scoped_lock lock(g_sessionMutex);
	g_sessions.insert(SessionMapElement(session));
}
void removeSession(const boost::shared_ptr<TcpSessionBase> &session){
	const boost::mutex::scoped_lock lock(g_sessionMutex);
	const AUTO(it, g_sessions.find<IDX_SESSION>(session));
	if(it == g_sessions.end<IDX_SESSION>()){
		LOG_ERROR("Socket not in epoll?");
		return;
	}
	g_sessions.erase<IDX_SESSION>(it);
}

void deepollReadable(const boost::shared_ptr<TcpSessionBase> &session){
	const unsigned long long now = getMonoClock();
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
	const unsigned long long now = getMonoClock();
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

void threadProc(){
	LOG_INFO("Epoll daemon started.");

	const AUTO(TCP_BUFFER_SIZE,
		ConfigFile::get<std::size_t>("tcp_buffer_size", 1024));
	const AUTO(MAX_EPOLL_TIMEOUT,
		ConfigFile::get<std::size_t>("max_epoll_timeout", 100));

	LOG_INFO("TCP_BUFFER_SIZE = ", TCP_BUFFER_SIZE,
		", MAX_EPOLL_TIMEOUT = ", MAX_EPOLL_TIMEOUT);

	const boost::scoped_array<unsigned char> data(new unsigned char[TCP_BUFFER_SIZE]);
	std::size_t epollTimeout = 0;

	std::vector<boost::shared_ptr<TcpSessionBase> > sessions;
	std::vector<boost::shared_ptr<const SocketServerBase> > servers;

	while(atomicLoad(g_daemonRunning)){
		// 第一部分，处理可接收的数据。
		{
			sessions.clear();
			const boost::mutex::scoped_lock lock(g_sessionMutex);
			for(AUTO(it, g_sessions.upperBound<IDX_READ>(0)); it != g_sessions.end<IDX_READ>(); ++it){
				sessions.push_back(it->m_session);
			}
		}
		for(AUTO(it, sessions.begin()); it != sessions.end(); ++it){
			epollTimeout = 0;
			const AUTO_REF(session, *it);
			try {
				const ::ssize_t bytesRead = ::recv(session->getFd(), data.get(), TCP_BUFFER_SIZE, MSG_NOSIGNAL);
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
				LOG_DEBUG("Read ", bytesRead, " byte(s) from ", session->getRemoteIp());
				session->onReadAvail(data.get(), bytesRead);
			} catch(Exception &e){
				LOG_ERROR("Exception thrown while dispatching data: file = ", e.file(),
					", line = ", e.line(), ", what = ", e.what());
				removeSession(session);
			} catch(std::exception &e){
				LOG_ERROR("std::exception thrown while dispatching data: what = ", e.what());
				removeSession(session);
			} catch(...){
				LOG_ERROR("Unknown exception thrown while dispatching data.");
				removeSession(session);
			}
		}

		// 第二部分，处理可发送的数据。
		{
			sessions.clear();
			const boost::mutex::scoped_lock lock(g_sessionMutex);
			for(AUTO(it, g_sessions.upperBound<IDX_WRITE>(0)); it != g_sessions.end<IDX_WRITE>(); ++it){
				sessions.push_back(it->m_session);
			}
		}
		for(AUTO(it, sessions.begin()); it != sessions.end(); ++it){
			epollTimeout = 0;
			const AUTO_REF(session, *it);
			try {
				std::size_t bytesToWrite;
				bool readShutdown;
				{
					boost::mutex::scoped_lock sessionLock;
					bytesToWrite = session->peekWriteAvail(sessionLock, data.get(), TCP_BUFFER_SIZE);
					readShutdown = session->hasReadBeenShutdown();
					if(bytesToWrite == 0){
						if(!readShutdown){
							reepollWriteable(session);
						}
					}
				}
				if(bytesToWrite == 0){
					if(readShutdown){
						removeSession(session);
					}
					continue;
				}
				const ::ssize_t bytesWritten = ::send(session->getFd(), data.get(), bytesToWrite, MSG_NOSIGNAL);
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
			} catch(Exception &e){
				LOG_ERROR("Exception thrown while writing socket: file = ", e.file(),
					", line = ", e.line(), ", what = ", e.what());
				removeSession(session);
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
		const int ready = ::epoll_wait(g_epoll.get(), events, COUNT_OF(events), epollTimeout);
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
					session->shutdownRead();
					event.events |= EPOLLIN;
					event.events |= EPOLLOUT;
				}
				if(event.events & EPOLLIN){
					deepollReadable(session);
				}
				if(event.events & EPOLLOUT){
					deepollWriteable(session);
				}
			} catch(Exception &e){
				LOG_ERROR("Exception thrown while epolling: file = ", e.file(),
					", line = ", e.line(), ", what = ", e.what());
				removeSession(session);
			} catch(std::exception &e){
				LOG_ERROR("std::exception thrown while epolling: what = ", e.what());
				removeSession(session);
			} catch(...){
				LOG_ERROR("Unknown exception thrown while epolling.");
				removeSession(session);
			}
		}

		// 二次指数回退算法。如果有连接接入（忙），epoll 等待时间就短一些；反之（闲）亦然。
		if(epollTimeout < MAX_EPOLL_TIMEOUT){
			epollTimeout = (epollTimeout << 1) | 1;
		}
	}

	LOG_INFO("Epoll daemon stopped.");
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

	g_sessions.clear();
	g_servers.clear();
}

void EpollDaemon::refreshSession(boost::shared_ptr<TcpSessionBase> session){
	assert(session);

	if(session->hasBeenShutdown()){
		return;
	}
	const unsigned long long now = getMonoClock();
	{
		const boost::mutex::scoped_lock lock(g_sessionMutex);
		const AUTO(it, g_sessions.find<0>(session));
		if(it == g_sessions.end<0>()){
			g_sessions.insert(SessionMapElement(session, now, now));
		} else {
			g_sessions.setKey<IDX_SESSION, IDX_READ>(it, now);
			g_sessions.setKey<IDX_SESSION, IDX_WRITE>(it, now);
		}
	}
}
void EpollDaemon::addSocketServer(boost::shared_ptr<SocketServerBase> server){
	assert(server);

	{
		const boost::mutex::scoped_lock lock(g_serverMutex);
		g_servers.insert(server);
	}
}
