#ifndef POSEIDON_TCP_SESSION_SSL_IMPL_
#	error Please do not #include "tcp_session_ssl_impl.hpp".
#endif

#ifndef POSEIDON_TCP_SESSION_SSL_IMPL_HPP_
#define POSEIDON_TCP_SESSION_SSL_IMPL_HPP_

#include <boost/noncopyable.hpp>
#include <openssl/ssl.h>
#include "raii.hpp"

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

	bool m_established;

public:
	SslImpl(Move<SslPtr> ssl, int fd);
	virtual ~SslImpl();

protected:
	::SSL *getSsl() const {
		return m_ssl.get();
	}
	virtual bool establishConnection() = 0;

public:
	long doRead(void *data, unsigned long size);
	long doWrite(const void *data, unsigned long size);
};

}

#endif
