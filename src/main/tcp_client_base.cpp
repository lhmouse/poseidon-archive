// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2015, LH_Mouse. All wrongs reserved.

#include "precompiled.hpp"
#include "tcp_client_base.hpp"
#include "tcp_session_base.hpp"
#include "ssl_factories.hpp"
#include "ssl_filter_base.hpp"
#include "sock_addr.hpp"
#include "ip_port.hpp"
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <errno.h>
#include <openssl/ssl.h>
#include "singletons/epoll_daemon.hpp"
#include "system_exception.hpp"
#include "log.hpp"

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
			const int ret = ::SSL_connect(getSsl());
			if(ret != 1){
				const int err = ::SSL_get_error(getSsl(), ret);
				if((err == SSL_ERROR_WANT_READ) || (err == SSL_ERROR_WANT_WRITE)){
					return false;
				}
				LOG_POSEIDON_ERROR("::SSL_connect() = ", ret, ", ::SSL_get_error() = ", err);
				DEBUG_THROW(Exception, SSLIT("::SSL_connect() failed"));
			}
			return true;
		}
	};

	const ClientSslFactory g_clientSslFactory;

	UniqueFile createSocket(int family){
		UniqueFile client(::socket(family, SOCK_STREAM, IPPROTO_TCP));
		if(!client){
			DEBUG_THROW(SystemException);
		}
		return client;
	}
}

TcpClientBase::TcpClientBase(const IpPort &addr, bool useSsl)
	: SockAddr(getSockAddrFromIpPort(addr)), TcpSessionBase(createSocket(SockAddr::getFamily()))
{
	if(::connect(m_socket.get(),
		static_cast<const ::sockaddr *>(SockAddr::getData()), SockAddr::getSize()) != 0)
	{
		if(errno != EINPROGRESS){
			DEBUG_THROW(SystemException);
		}
	}
	if(useSsl){
		LOG_POSEIDON_INFO("Initiating SSL handshake...");

		AUTO(ssl, g_clientSslFactory.createSsl());
		boost::scoped_ptr<SslFilterBase> filter(new SslFilter(STD_MOVE(ssl), getFd()));
		initSsl(STD_MOVE(filter));
	}
}
TcpClientBase::~TcpClientBase(){
}

void TcpClientBase::goResident(){
	EpollDaemon::addSession(virtualSharedFromThis<TcpSessionBase>());
}

}
