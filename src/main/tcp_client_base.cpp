#include "../precompiled.hpp"
#include "tcp_client_base.hpp"
#include "tcp_session_base.hpp"
#define POSEIDON_TCP_SESSION_SSL_IMPL_
#include "tcp_session_ssl_impl.hpp"
#include <arpa/inet.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <errno.h>
#include <endian.h>
#include <openssl/ssl.h>
#include "singletons/epoll_daemon.hpp"
#include "exception.hpp"
#include "log.hpp"
using namespace Poseidon;

namespace {

class ClientSslCtx {
private:
	const SslCtxPtr m_sslCtx;

public:
	ClientSslCtx()
		: m_sslCtx(::SSL_CTX_new(::SSLv23_client_method()))
	{
		::SSL_CTX_set_verify(m_sslCtx.get(), SSL_VERIFY_NONE, NULLPTR);
	}

public:
	void createSsl(SslPtr &ssl){
		ssl.reset(::SSL_new(m_sslCtx.get()));
	}
} g_clientSslCtx;

}

class TcpClientBase::SslImplClient : public TcpSessionBase::SslImpl {
public:
	SslImplClient(Move<SslPtr> ssl, int fd)
		: SslImpl(STD_MOVE(ssl), fd)
	{
	}

protected:
	bool establishConnection(){
		const int ret = ::SSL_connect(m_ssl.get());
		if(ret == 1){
			return true;
		}
		const int err = ::SSL_get_error(m_ssl.get(), ret);
		if((err == SSL_ERROR_WANT_READ) || (err == SSL_ERROR_WANT_WRITE)){
			return false;
		}
		LOG_ERROR("::SSL_connect() = ", ret, ", ::SSL_get_error() = ", err);
		DEBUG_THROW(Exception, "::SSL_connect() failed");
	}
};

void TcpClientBase::connect(ScopedFile &client, const std::string &ip, unsigned port){
	union {
		::sockaddr sa;
		::sockaddr_in sin;
		::sockaddr_in6 sin6;
	} u;
	::socklen_t salen = sizeof(u);

	if(::inet_pton(AF_INET, ip.c_str(), &u.sin.sin_addr) == 1){
		u.sin.sin_family = AF_INET;
		u.sin.sin_port = be16toh(port);
		salen = sizeof(::sockaddr_in);
	} else if(::inet_pton(AF_INET6, ip.c_str(), &u.sin6.sin6_addr) == 1){
		u.sin6.sin6_family = AF_INET6;
		u.sin6.sin6_port = be16toh(port);
		salen = sizeof(::sockaddr_in6);
	} else {
		LOG_ERROR("Unknown address format: ", ip);
		DEBUG_THROW(Exception, "Unknown address format");
	}

	client.reset(::socket(u.sa.sa_family, SOCK_STREAM, IPPROTO_TCP));
	if(!client){
		DEBUG_THROW(SystemError, errno);
	}
	if(::connect(client.get(), &u.sa, salen) != 0){
		DEBUG_THROW(SystemError, errno);
	}
}

TcpClientBase::TcpClientBase(Move<ScopedFile> socket)
	: TcpSessionBase(STD_MOVE(socket))
{
}

void TcpClientBase::sslConnect(){
	LOG_INFO("Initiating SSL handshake...");

	SslPtr ssl;
	g_clientSslCtx.createSsl(ssl);
	boost::scoped_ptr<TcpSessionBase::SslImpl>
		sslImpl(new SslImplClient(STD_MOVE(ssl), m_socket.get()));
	initSsl(STD_MOVE(sslImpl));
}
void TcpClientBase::goResident(){
	EpollDaemon::addSession(virtualSharedFromThis<TcpSessionBase>());
}
