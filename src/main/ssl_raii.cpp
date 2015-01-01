// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2015, LH_Mouse. All wrongs reserved.

#include "precompiled.hpp"
#include "ssl_raii.hpp"
#include <openssl/ssl.h>
using namespace Poseidon;

void SslDeleter::operator()(::SSL *ssl) NOEXCEPT {
	::SSL_free(ssl);
}

void SslCtxDeleter::operator()(::SSL_CTX *sslCtx) NOEXCEPT {
	::SSL_CTX_free(sslCtx);
}
