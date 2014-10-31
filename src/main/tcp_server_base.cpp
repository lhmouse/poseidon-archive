#include "precompiled.hpp"
#include "tcp_server_base.hpp"
#include "tcp_session_base.hpp"
#define POSEIDON_TCP_SESSION_SSL_IMPL_
#include "tcp_session_ssl_impl.hpp"
#include <arpa/inet.h>
#include <fcntl.h>
#include <errno.h>
#include "singletons/epoll_daemon.hpp"
#include "log.hpp"
#include "exception.hpp"
#include "endian.hpp"
#include "tcp_session_base.hpp"
using namespace Poseidon;

class TcpServerBase::SslImplServer : boost::noncopyable {
private:
	SslCtxPtr m_sslCtx;

public:
	SslImplServer(const SharedNtmbs &cert, const SharedNtmbs &privateKey){
		requireSsl();

		m_sslCtx.reset(::SSL_CTX_new(::TLSv1_server_method()));
		if(!m_sslCtx){
			LOG_POSEIDON_FATAL("Could not create server SSL context");
			std::abort();
		}
		::SSL_CTX_set_verify(m_sslCtx.get(), SSL_VERIFY_PEER | SSL_VERIFY_CLIENT_ONCE, VAL_INIT);

		LOG_POSEIDON_INFO("Loading server certificate: ", cert);
		if(::SSL_CTX_use_certificate_file(m_sslCtx.get(), cert.get(), SSL_FILETYPE_PEM) != 1){
			DEBUG_THROW(Exception, "::SSL_CTX_use_certificate_file() failed");
		}

		LOG_POSEIDON_INFO("Loading server private key: ", privateKey);
		if(::SSL_CTX_use_PrivateKey_file(m_sslCtx.get(), privateKey.get(), SSL_FILETYPE_PEM) != 1){
			DEBUG_THROW(Exception, "::SSL_CTX_use_PrivateKey_file() failed");
		}

		LOG_POSEIDON_INFO("Verifying private key...");
		if(::SSL_CTX_check_private_key(m_sslCtx.get()) != 1){
			DEBUG_THROW(Exception, "::SSL_CTX_check_private_key() failed");
		}
	}

public:
	SslPtr createSsl() const {
		return SslPtr(::SSL_new(m_sslCtx.get()));
	}
};

class TcpServerBase::SslImplClient : public TcpSessionBase::SslImpl {
public:
	SslImplClient(SslPtr ssl, int fd)
		: SslImpl(STD_MOVE(ssl), fd)
	{
	}

protected:
	bool establishConnection(){
		const int ret = ::SSL_accept(m_ssl.get());
		if(ret != 1){
			const int err = ::SSL_get_error(m_ssl.get(), ret);
			if((err == SSL_ERROR_WANT_READ) || (err == SSL_ERROR_WANT_WRITE)){
				return false;
			}
			LOG_POSEIDON_ERROR("::SSL_accept() = ", ret, ", ::SSL_get_error() = ", err);
			DEBUG_THROW(Exception, "::SSL_accept() failed");
		}
		return true;
	}
};

TcpServerBase::TcpServerBase(std::string bindAddr, unsigned bindPort,
	const SharedNtmbs &cert, const SharedNtmbs &privateKey)
	: m_bindAddr(STD_MOVE(bindAddr)), m_bindPort(bindPort)
{
	union {
		::sockaddr sa;
		::sockaddr_in sin;
		::sockaddr_in6 sin6;
	} u;
	::socklen_t salen = sizeof(u);

	if(::inet_pton(AF_INET, m_bindAddr.c_str(), &u.sin.sin_addr) == 1){
		u.sin.sin_family = AF_INET;
		storeBe(u.sin.sin_port, m_bindPort);
		salen = sizeof(::sockaddr_in);
	} else if(::inet_pton(AF_INET6, m_bindAddr.c_str(), &u.sin6.sin6_addr) == 1){
		u.sin6.sin6_family = AF_INET6;
		storeBe(u.sin6.sin6_port, m_bindPort);
		salen = sizeof(::sockaddr_in6);
	} else {
		LOG_POSEIDON_ERROR("Unknown address format: ", m_bindAddr);
		DEBUG_THROW(Exception, "Unknown address format");
	}

	m_listen.reset(::socket(u.sa.sa_family, SOCK_STREAM, IPPROTO_TCP));
	if(!m_listen){
		DEBUG_THROW(SystemError);
	}
	const int TRUE_VALUE = true;
	if(::setsockopt(m_listen.get(),
		SOL_SOCKET, SO_REUSEADDR, &TRUE_VALUE, sizeof(TRUE_VALUE)) != 0)
	{
		DEBUG_THROW(SystemError);
	}
	const int flags = ::fcntl(m_listen.get(), F_GETFL);
	if(flags == -1){
		DEBUG_THROW(SystemError);
	}
	if(::fcntl(m_listen.get(), F_SETFL, flags | O_NONBLOCK) != 0){
		DEBUG_THROW(SystemError);
	}
	if(::bind(m_listen.get(), &u.sa, salen)){
		DEBUG_THROW(SystemError);
	}
	if(::listen(m_listen.get(), SOMAXCONN)){
		DEBUG_THROW(SystemError);
	}

	if(cert){
		m_sslImplServer.reset(new SslImplServer(cert, privateKey));
	}

	LOG_POSEIDON_INFO("Created ", (m_sslImplServer ? "SSL " : ""), "socket server on ", m_bindAddr, ':', m_bindPort);
}
TcpServerBase::~TcpServerBase(){
	LOG_POSEIDON_INFO("Destroyed ", (m_sslImplServer ? "SSL " : ""), "socket server on ", m_bindAddr, ':', m_bindPort);
}

boost::shared_ptr<TcpSessionBase> TcpServerBase::tryAccept() const {
	ScopedFile client(::accept(m_listen.get(), VAL_INIT, VAL_INIT));
	if(!client){
		if(errno != EAGAIN){
			DEBUG_THROW(SystemError);
		}
		return VAL_INIT;
	}
	AUTO(session, onClientConnect(STD_MOVE(client)));
	if(!session){
		LOG_POSEIDON_WARN("onClientConnect() returns a null pointer.");
		DEBUG_THROW(Exception, "Null client pointer");
	}
	if(m_sslImplServer){
		LOG_POSEIDON_INFO("Waiting for SSL handshake...");

		boost::scoped_ptr<TcpSessionBase::SslImpl> sslImpl(
			new SslImplClient(m_sslImplServer->createSsl(), session->m_socket.get()));
		session->initSsl(STD_MOVE(sslImpl));
	}
	LOG_POSEIDON_INFO("Client connected from ",
		session->getRemoteIp(), ':', session->getRemotePort());
	return session;
}
