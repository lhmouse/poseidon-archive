// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2015, LH_Mouse. All wrongs reserved.

#include "precompiled.hpp"
#include "tcp_server_base.hpp"
#include "tcp_session_base.hpp"
#include "ssl_factories.hpp"
#include "ssl_filter_base.hpp"
#include "sock_addr.hpp"
#include "ip_port.hpp"
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <openssl/ssl.h>
#include "singletons/epoll_daemon.hpp"
#include "log.hpp"
#include "exception.hpp"
#include "endian.hpp"

namespace Poseidon {

namespace {
	class SslFilter : public SslFilterBase {
	public:
		SslFilter(Move<UniqueSsl> ssl, int fd)
			: SslFilterBase(STD_MOVE(ssl), fd)
		{
		}

	private:
		bool establish(){
			const int ret = ::SSL_accept(getSsl());
			if(ret != 1){
				const int err = ::SSL_get_error(getSsl(), ret);
				if((err == SSL_ERROR_WANT_READ) || (err == SSL_ERROR_WANT_WRITE)){
					return false;
				}
				LOG_POSEIDON_ERROR("::SSL_accept() = ", ret, ", ::SSL_get_error() = ", err);
				DEBUG_THROW(Exception, SharedNts::observe("::SSL_accept() failed"));
			}
			return true;
		}
	};

	UniqueFile createListenSocket(const IpPort &addr){
		SockAddr sa = getSockAddrFromIpPort(addr);
		UniqueFile listen(::socket(sa.getFamily(), SOCK_STREAM, IPPROTO_TCP));
		if(!listen){
			DEBUG_THROW(SystemError);
		}
		const int TRUE_VALUE = true;
		if(::setsockopt(listen.get(), SOL_SOCKET, SO_REUSEADDR, &TRUE_VALUE, sizeof(TRUE_VALUE)) != 0){
			DEBUG_THROW(SystemError);
		}
		if(::bind(listen.get(), static_cast<const ::sockaddr *>(sa.getData()), sa.getSize())){
			DEBUG_THROW(SystemError);
		}
		if(::listen(listen.get(), SOMAXCONN)){
			DEBUG_THROW(SystemError);
		}
		return listen;
	}
}

TcpServerBase::TcpServerBase(const IpPort &bindAddr, const char *cert, const char *privateKey)
	: SocketServerBase(createListenSocket(bindAddr))
{
	if(cert && (cert[0] != 0)){
		m_sslFactory.reset(new ServerSslFactory(cert, privateKey));
	}

	LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_INFO, "Created ", (m_sslFactory ? "SSL " : ""), "TCP server on ", getLocalInfo());
}
TcpServerBase::~TcpServerBase(){
	LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_INFO, "Destroyed ", (m_sslFactory ? "SSL " : ""), "TCP server on ", getLocalInfo());
}

bool TcpServerBase::poll() const {
	UniqueFile client(::accept(getFd(), NULLPTR, NULLPTR));
	if(!client){
		if(errno != EAGAIN){
			DEBUG_THROW(SystemError);
		}
		return false;
	}
	AUTO(session, onClientConnect(STD_MOVE(client)));
	if(!session){
		LOG_POSEIDON_WARNING("onClientConnect() returns a null pointer.");
		DEBUG_THROW(Exception, SharedNts::observe("Null client pointer"));
	}
	if(m_sslFactory){
		AUTO(ssl, m_sslFactory->createSsl());
		boost::scoped_ptr<SslFilterBase> filter(new SslFilter(STD_MOVE(ssl), session->getFd()));
		session->initSsl(STD_MOVE(filter));
	}
	session->setTimeout(EpollDaemon::getTcpRequestTimeout());
	EpollDaemon::addSession(session);
	LOG_POSEIDON_INFO("Accepted TCP connection from ", session->getRemoteInfo());
	return true;
}

}
