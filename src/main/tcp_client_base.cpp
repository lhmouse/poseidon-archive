#include "precompiled.hpp"
#include "tcp_client_base.hpp"
#include "tcp_session_base.hpp"
#define POSEIDON_TCP_SESSION_SSL_IMPL_
#include "tcp_session_ssl_impl.hpp"
#include <arpa/inet.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <errno.h>
#include <openssl/ssl.h>
#include "singletons/epoll_daemon.hpp"
#include "exception.hpp"
#include "log.hpp"
#include "endian.hpp"
using namespace Poseidon;

namespace {

class ClientSslCtx : boost::noncopyable {
private:
	SslCtxPtr m_sslCtx;

public:
	ClientSslCtx(){
		requireSsl();

		m_sslCtx.reset(::SSL_CTX_new(::TLSv1_client_method()));
		if(!m_sslCtx){
			LOG_POSEIDON_FATAL("Could not create client SSL context");
			std::abort();
		}
		::SSL_CTX_set_verify(m_sslCtx.get(), SSL_VERIFY_NONE, VAL_INIT);
	}

public:
	SslPtr createSsl() const {
		return SslPtr(::SSL_new(m_sslCtx.get()));
	}
} g_clientSslCtx;

ScopedFile parseAddrPort(void *sa, unsigned &salen, unsigned maxSalen, const IpPort &addr){
	union {
		::sockaddr sa;
		::sockaddr_in sin;
		::sockaddr_in6 sin6;
	} u;

	if(::inet_pton(AF_INET, addr.ip.get(), &u.sin.sin_addr) == 1){
		u.sin.sin_family = AF_INET;
		storeBe(u.sin.sin_port, addr.port);
		salen = sizeof(::sockaddr_in);
	} else if(::inet_pton(AF_INET6, addr.ip.get(), &u.sin6.sin6_addr) == 1){
		u.sin6.sin6_family = AF_INET6;
		storeBe(u.sin6.sin6_port, addr.port);
		salen = sizeof(::sockaddr_in6);
	} else {
		LOG_POSEIDON_ERROR("Unknown address format: ", addr.ip);
		DEBUG_THROW(Exception, "Unknown address format");
	}
	if(maxSalen < salen){
		LOG_POSEIDON_ERROR("Buffer for sa is too small: ",
			maxSalen, " is provided but ", salen, " is required.");
	}
	std::memcpy(sa, &u, salen);

	ScopedFile client(::socket(u.sa.sa_family, SOCK_STREAM, IPPROTO_TCP));
	if(!client){
		DEBUG_THROW(SystemError);
	}
	return client;
}

}

class TcpClientBase::SslImplClient : public TcpSessionBase::SslImpl {
public:
	SslImplClient(SslPtr ssl, int fd)
		: SslImpl(STD_MOVE(ssl), fd)
	{
	}

protected:
	bool establishConnection(){
		const int ret = ::SSL_connect(m_ssl.get());
		if(ret != 1){
			const int err = ::SSL_get_error(m_ssl.get(), ret);
			if((err == SSL_ERROR_WANT_READ) || (err == SSL_ERROR_WANT_WRITE)){
				return false;
			}
			LOG_POSEIDON_ERROR("::SSL_connect() = ", ret, ", ::SSL_get_error() = ", err);
			DEBUG_THROW(Exception, "::SSL_connect() failed");
		}
		return true;
	}
};

TcpClientBase::TcpClientBase(const IpPort &addr)
	: TcpSessionBase(parseAddrPort(m_sa, m_salen, sizeof(m_sa), addr))
{
	if(::connect(m_socket.get(), reinterpret_cast<const ::sockaddr *>(m_sa), m_salen) != 0){
		if(errno != EINPROGRESS){
			DEBUG_THROW(SystemError);
		}
	}
}

void TcpClientBase::sslConnect(){
	LOG_POSEIDON_INFO("Initiating SSL handshake...");

	boost::scoped_ptr<TcpSessionBase::SslImpl> sslImpl(
		new SslImplClient(g_clientSslCtx.createSsl(), m_socket.get()));
	initSsl(STD_MOVE(sslImpl));
}
void TcpClientBase::goResident(){
	EpollDaemon::addSession(virtualSharedFromThis<TcpSessionBase>());
}
