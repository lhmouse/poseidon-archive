// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014, LH_Mouse. All wrongs reserved.

#include "precompiled.hpp"
#include "tcp_client_base.hpp"
#include "tcp_session_base.hpp"
#define POSEIDON_TCP_SESSION_SSL_IMPL_
#include "tcp_session_ssl_impl.hpp"
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <errno.h>
#include <openssl/ssl.h>
#include "sock_addr.hpp"
#include "ip_port.hpp"
#include "singletons/epoll_daemon.hpp"
#include "exception.hpp"
#include "log.hpp"
using namespace Poseidon;

namespace {

class ClientSslCtx : boost::noncopyable {
private:
	SslCtxPtr m_sslCtx;

public:
	ClientSslCtx(){
		requireSsl();

		if(!m_sslCtx.reset(::SSL_CTX_new(::TLSv1_client_method()))){
			LOG_POSEIDON_FATAL("Could not create client SSL context");
			std::abort();
		}
		::SSL_CTX_set_verify(m_sslCtx.get(), SSL_VERIFY_NONE, NULLPTR);
	}

public:
	SslPtr createSsl() const {
		return SslPtr(::SSL_new(m_sslCtx.get()));
	}
} g_clientSslCtx;

ScopedFile createSocket(SockAddr &sa, const IpPort &addr){
	sa = getSockAddrFromIpPort(addr);
	ScopedFile client(::socket(sa.getFamily(), SOCK_STREAM, IPPROTO_TCP));
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
	: TcpSessionBase(createSocket(m_sockAddr, addr))
{
	if(::connect(m_socket.get(),
		static_cast<const ::sockaddr *>(m_sockAddr.getData()), m_sockAddr.getSize()) != 0)
	{
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
