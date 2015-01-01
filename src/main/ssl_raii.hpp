// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2015, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_SSL_RAII_HPP_
#define POSEIDON_SSL_RAII_HPP_

#include "cxx_ver.hpp"
#include <openssl/ossl_typ.h>
#include "raii.hpp"

namespace Poseidon {

struct SslDeleter {
	CONSTEXPR ::SSL *operator()() NOEXCEPT {
		return NULLPTR;
	}
	void operator()(::SSL *ssl) NOEXCEPT;
};

typedef UniqueHandle<SslDeleter> UniqueSsl;

struct SslCtxDeleter {
	CONSTEXPR ::SSL_CTX *operator()() NOEXCEPT {
		return NULLPTR;
	}
	void operator()(::SSL_CTX *sslCtx) NOEXCEPT;
};

typedef UniqueHandle<SslCtxDeleter> UniqueSslCtx;

}

#endif
