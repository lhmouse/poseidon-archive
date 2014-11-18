// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_TCP_SESSION_SSL_IMPL_
#	error Please do not #include "tcp_session_ssl_impl.hpp".
#endif

#ifndef POSEIDON_TCP_SESSION_SSL_IMPL_HPP_
#define POSEIDON_TCP_SESSION_SSL_IMPL_HPP_

#include <boost/noncopyable.hpp>
#include <openssl/ssl.h>
#include "raii.hpp"

namespace Poseidon {

extern void requireSsl();

struct SslCtxDeleter {
	CONSTEXPR ::SSL_CTX *operator()() NOEXCEPT {
		return VAL_INIT;
	}
	void operator()(::SSL_CTX *ctx) NOEXCEPT {
		::SSL_CTX_free(ctx);
	}
};
typedef ScopedHandle<SslCtxDeleter> SslCtxPtr;

struct SslDeleter {
	CONSTEXPR ::SSL *operator()() NOEXCEPT {
		return VAL_INIT;
	}
	void operator()(::SSL *ssl) NOEXCEPT {
		::SSL_free(ssl);
	}
};
typedef ScopedHandle<SslDeleter> SslPtr;

class TcpSessionBase::SslImpl : boost::noncopyable {
protected:
	const SslPtr m_ssl;

	bool m_established;

public:
	SslImpl(SslPtr ssl, int fd);
	virtual ~SslImpl();

protected:
	virtual bool establishConnection() = 0;

public:
	long doRead(void *data, unsigned long size);
	long doWrite(const void *data, unsigned long size);
};

}

#endif
