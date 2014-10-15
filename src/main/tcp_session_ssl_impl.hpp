#ifndef POSEIDON_TCP_SESSION_BASE_HPP_
#	error Please do not #include "tcp_session_ssl_impl.hpp".
#endif

#ifndef POSEIDON_TCP_SESSION_SSL_IMPL_HPP_
#define POSEIDON_TCP_SESSION_SSL_IMPL_HPP_

#include <openssl/ssl.h>
#include "raii.hpp"
#include "exception.hpp"

namespace Poseidon {

struct SslCtxDeleter {
	CONSTEXPR ::SSL_CTX *operator()() NOEXCEPT {
		return NULLPTR;
	}
	void operator()(::SSL_CTX *ctx) NOEXCEPT {
		::SSL_CTX_free(ctx);
	}
};
typedef ScopedHandle<SslCtxDeleter> SslCtxPtr;

struct SslDeleter {
	CONSTEXPR ::SSL *operator()() NOEXCEPT {
		return NULLPTR;
	}
	void operator()(::SSL *ssl) NOEXCEPT {
		::SSL_free(ssl);
	}
};
typedef ScopedHandle<SslDeleter> SslPtr;

class TcpSessionBase::SslImpl : boost::noncopyable {
private:
	const SslPtr m_ssl;

public:
	SslImpl(Move<SslPtr> ssl, int fd)
		: m_ssl(STD_MOVE(ssl))
	{
		if(!::SSL_set_fd(m_ssl.get(), fd)){
			DEBUG_THROW(Exception, "::SSL_set_fd() failed");
		}
	}
	~SslImpl(){
		::SSL_shutdown(m_ssl.get());
	}

public:
	// Client
	void connect(){
		if(::SSL_connect(m_ssl.get()) != 1){
			DEBUG_THROW(Exception, "::SSL_connect() failed");
		}
	}
	// Server
	void accept(){
		if(::SSL_accept(m_ssl.get()) != 1){
			DEBUG_THROW(Exception, "::SSL_accept() failed");
		}
	}

	long doRead(void *data, unsigned long size){
		return ::SSL_read(m_ssl.get(), data, size);
	}
	long doWrite(const void *data, unsigned long size){
		return ::SSL_write(m_ssl.get(), data, size);
	}
};

}

#endif
