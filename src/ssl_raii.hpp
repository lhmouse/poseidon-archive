// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2018, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_SSL_RAII_HPP_
#define POSEIDON_SSL_RAII_HPP_

#include "cxx_ver.hpp"
#include "raii.hpp"
#include <openssl/ssl.h>

namespace Poseidon {

struct Ssl_deleter {
	CONSTEXPR ::SSL *operator()() NOEXCEPT {
		return NULLPTR;
	}
	void operator()(::SSL *ssl) NOEXCEPT;
};

typedef Unique_handle<Ssl_deleter> Unique_ssl;

struct Ssl_ctx_deleter {
	CONSTEXPR ::SSL_CTX *operator()() NOEXCEPT {
		return NULLPTR;
	}
	void operator()(::SSL_CTX *ssl_ctx) NOEXCEPT;
};

typedef Unique_handle<Ssl_ctx_deleter> Unique_ssl_ctx;

}

#endif
