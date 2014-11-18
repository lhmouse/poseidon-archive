// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014, LH_Mouse. All wrongs reserved.

#include "precompiled.hpp"
#include "tcp_client_base.hpp"
#include "tcp_session_base.hpp"
#define POSEIDON_TCP_SESSION_SSL_IMPL_
#include "tcp_session_ssl_impl.hpp"
#define POSEIDON_SOCK_ADDR_
#include "sock_addr.hpp"
#include <boost/static_assert.hpp>
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

ScopedFile createSocket(SockAddr &sa, unsigned &salen, const IpPort &addr){
	sa = getSockAddrFromIpPort(salen, addr);

	ScopedFile client(::socket(sa.sa.sa_family, SOCK_STREAM, IPPROTO_TCP));
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
	: TcpSessionBase(createSocket(reinterpret_cast<SockAddr &>(m_sa), m_salen, addr))
{
	BOOST_STATIC_ASSERT(sizeof(m_sa) >= sizeof(SockAddr));

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
