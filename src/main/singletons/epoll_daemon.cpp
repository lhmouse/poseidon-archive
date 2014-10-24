#include "../../precompiled.hpp"
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
#define POSEIDON_TCP_SESSION_IMPL_
#include "../tcp_session_base.hpp"
#include "../tcp_session_impl.hpp"
#include "../multi_index_map.hpp"
#include "../profiler.hpp"
#include "../player/server.hpp"
#include "../http/server.hpp"
using namespace Poseidon;

namespace {

std::size_t g_dataBufferSize	= 1024;
std::size_t g_eventBufferSize	= 256;
std::size_t g_maxTimeout		= 100;

volatile bool g_running = false;
ScopedFile g_epoll;
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
};

MULTI_INDEX_MAP(SessionMap, SessionMapElement,
	UNIQUE_MEMBER_INDEX(session),
	MULTI_MEMBER_INDEX(lastRead),
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
std::list<boost::weak_ptr<const TcpServerBase> > g_servers;

void add(const boost::shared_ptr<TcpSessionBase> &session){
	const AUTO(now, getMonoClock());
	std::pair<SessionMap::iterator, bool> result;
	{
		const boost::mutex::scoped_lock lock(g_sessionMutex);
		result = g_sessions.insert(SessionMapElement(session, now, now));
		if(!result.second){
			LOG_WARN("Socket already in epoll?");
			return;
		}
	}
	::epoll_event event;
	event.events = EPOLLIN | EPOLLOUT | EPOLLET;
	event.data.ptr = session.get();
	if(::epoll_ctl(g_epoll.get(), EPOLL_CTL_ADD,
		TcpSessionImpl::doGetFd(*session), &event) != 0)
	{
		const boost::mutex::scoped_lock lock(g_sessionMutex);
		g_sessions.erase(result.first);
		DEBUG_THROW(SystemError, errno);
	}
}
void touch(const boost::shared_ptr<TcpSessionBase> &session){
	const AUTO(now, getMonoClock());
	const boost::mutex::scoped_lock lock(g_sessionMutex);
	const AUTO(it, g_sessions.find<0>(session));
	if(it == g_sessions.end<0>()){
		LOG_WARN("Socket not in epoll?");
		return;
	}
	g_sessions.setKey<IDX_SESSION, IDX_READ>(it, now);
	g_sessions.setKey<IDX_SESSION, IDX_WRITE>(it, now);
}
void remove(const boost::shared_ptr<TcpSessionBase> &session){
	if(::epoll_ctl(g_epoll.get(), EPOLL_CTL_DEL,
		TcpSessionImpl::doGetFd(*session), VAL_INIT) != 0)
	{
		LOG_WARN("Error deleting from epoll. We can do nothing but ignore it.");
	}
	const boost::mutex::scoped_lock lock(g_sessionMutex);
	const AUTO(it, g_sessions.find<IDX_SESSION>(session));
	if(it == g_sessions.end<IDX_SESSION>()){
		LOG_WARN("Socket not in epoll?");
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
	const boost::scoped_array<unsigned char> data(new unsigned char[g_dataBufferSize]);
	const boost::scoped_array< ::epoll_event> events(new ::epoll_event[g_eventBufferSize]);

	std::size_t epollTimeout = 0;
	std::vector<boost::shared_ptr<TcpSessionBase> > sessions;
	std::vector<boost::shared_ptr<const TcpServerBase> > servers;

	while(atomicLoad(g_running)){
		// 第一部分，处理可接收的数据。
		{
			const boost::mutex::scoped_lock lock(g_sessionMutex);
			for(AUTO(it, g_sessions.upperBound<IDX_READ>(0));
				it != g_sessions.end<IDX_READ>(); ++it)
			{
				sessions.push_back(it->session);
			}
		}
		for(AUTO(it, sessions.begin()); it != sessions.end(); ++it){
			const AUTO_REF(session, *it);
			try {
				if(session->hasBeenShutdown()){
					continue;
				}
				const ::ssize_t bytesRead = TcpSessionImpl::doRead(*session,
					data.get(), g_dataBufferSize);
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
					LOG_INFO("Connection closed by remote host: ", session->getRemoteIp());
					session->send(StreamBuffer(), true);
					continue;
				}
				LOG_TRACE("Read ", bytesRead, " byte(s) from ", session->getRemoteIp(),
					": ", HexEncoder(data.get(), bytesRead));
			} catch(std::exception &e){
				LOG_ERROR("std::exception thrown while dispatching data: what = ", e.what());
				session->send(StreamBuffer(), true);
			} catch(...){
				LOG_ERROR("Unknown exception thrown while dispatching data.");
				session->send(StreamBuffer(), true);
			}
			epollTimeout = 0;
		}
		sessions.clear();

		// 第二部分，处理可发送的数据。
		{
			const boost::mutex::scoped_lock lock(g_sessionMutex);
			for(AUTO(it, g_sessions.upperBound<IDX_WRITE>(0));
				it != g_sessions.end<IDX_WRITE>(); ++it)
			{
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
					bytesWritten = TcpSessionImpl::doWrite(*session,
						sessionLock, data.get(), g_dataBufferSize);
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
					DEBUG_THROW(SystemError, errno);
				}
				if(bytesWritten == 0){
					if(shutdown){
						remove(session);
					}
					continue;
				}
				LOG_TRACE("Wrote ", bytesWritten, " byte(s) to ", session->getRemoteIp(),
					": ", HexEncoder(data.get(), bytesWritten));
			} catch(std::exception &e){
				LOG_ERROR("std::exception thrown while writing socket: what = ", e.what());
				remove(session);
			} catch(...){
				LOG_ERROR("Unknown exception thrown while writing socket.");
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
				AUTO(session, (*it)->tryAccept());
				if(!session){
					continue;
				}
				LOG_DEBUG("Accepted socket connection from ", session->getRemoteIp());
				add(session);
			} catch(std::exception &e){
				LOG_ERROR("std::exception thrown while accepting client: what = ", e.what());
			} catch(...){
				LOG_ERROR("Unknown exception thrown while accepting client.");
			}
			epollTimeout = 0;
		}
		servers.clear();

		// 第四部分，检测新的数据。
		const int ready = ::epoll_wait(g_epoll.get(), events.get(), g_eventBufferSize, epollTimeout);
		if(ready < 0){
			const AUTO(desc, getErrorDesc());
			LOG_ERROR("::epoll_wait() failed: ", desc);
		} else for(unsigned i = 0; i < (unsigned)ready; ++i){
			::epoll_event &event = events[i];
			const AUTO(session, static_cast<TcpSessionBase *>(event.data.ptr)->
				virtualSharedFromThis<TcpSessionBase>());
			try {
				if(event.events & EPOLLHUP){
					LOG_INFO("Socket hung up, ip = ", session->getRemoteIp());
					remove(session);
					continue;
				}
				if(event.events & EPOLLERR){
					int err;
					::socklen_t errLen = sizeof(err);
					if(::getsockopt(TcpSessionImpl::doGetFd(*session),
						SOL_SOCKET, SO_ERROR, &err, &errLen) != 0)
					{
						err = errno;
					}
					const AUTO(desc, getErrorDesc());
					LOG_WARN("Socket error: ", desc);
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
				LOG_ERROR("std::exception thrown while epolling: what = ", e.what());
				remove(session);
			} catch(...){
				LOG_ERROR("Unknown exception thrown while epolling.");
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
		LOG_DEBUG("Flushing data on ", remaining.size(), " socket(s).");

		for(AUTO(it, remaining.begin()); it != remaining.end(); ++it){
			const AUTO_REF(session, it->session);
			try {
				::ssize_t bytesWritten;
				for(;;){
					{
						boost::mutex::scoped_lock sessionLock;
						bytesWritten = TcpSessionImpl::doWrite(*session,
							sessionLock, data.get(), g_dataBufferSize);
					}
					if(bytesWritten <= 0){
						break;
					}
					LOG_TRACE("Flushed ", bytesWritten, " byte(s) to ", session->getRemoteIp(),
						": ", HexEncoder(data.get(), bytesWritten));
				}
			} catch(std::exception &e){
				LOG_ERROR("std::exception thrown while flush data: what = ", e.what());
			} catch(...){
				LOG_ERROR("Unknown exception thrown while flush data.");
			}
		}
	}
}

void threadProc(){
	PROFILE_ME;
	Logger::setThreadTag("   N"); // Network
	LOG_INFO("Epoll daemon started.");

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

	MainConfig::get(g_dataBufferSize, "epoll_data_buffer_size");
	LOG_DEBUG("Data buffer size = ", g_dataBufferSize);

	MainConfig::get(g_eventBufferSize, "epoll_event_buffer_size");
	LOG_DEBUG("Event buffer size = ", g_eventBufferSize);

	MainConfig::get(g_maxTimeout, "epoll_max_timeout");
	LOG_DEBUG("Max timeout = ", g_maxTimeout);

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

std::vector<EpollSnapshotItem> EpollDaemon::snapshot(){
	std::vector<EpollSnapshotItem> ret;
	const AUTO(now, getMonoClock());
	const boost::mutex::scoped_lock lock(g_sessionMutex);
	ret.reserve(g_sessions.size());
	for(AUTO(it, g_sessions.begin()); it != g_sessions.end(); ++it){
		ret.push_back(VAL_INIT);
		AUTO_REF(item, ret.back());
		item.remoteIp = it->session->getRemoteIp();
		item.remotePort = it->session->getRemotePort();
		item.localIp = it->session->getLocalIp();
		item.localPort  = it->session->getLocalPort();
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

boost::shared_ptr<PlayerServer> EpollDaemon::registerPlayerServer(
	const std::string &bindAddr, unsigned bindPort,
	const std::string &cert, const std::string &privateKey)
{
	AUTO(newServer, boost::make_shared<PlayerServer>(bindAddr, bindPort, cert, privateKey));
	{
		const boost::mutex::scoped_lock lock(g_serverMutex);
		g_servers.push_back(newServer);
	}
	return newServer;
}
boost::shared_ptr<HttpServer> EpollDaemon::registerHttpServer(
	const std::string &bindAddr, unsigned bindPort,
	const std::string &cert, const std::string &privateKey, const std::vector<std::string> &auth)
{
	AUTO(newServer, boost::make_shared<HttpServer>(bindAddr, bindPort, cert, privateKey, auth));
	{
		const boost::mutex::scoped_lock lock(g_serverMutex);
		g_servers.push_back(newServer);
	}
	return newServer;
}
