// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2015, LH_Mouse. All wrongs reserved.

#include "precompiled.hpp"
#include "ssl_factories.hpp"
#include <pthread.h>
#include <errno.h>
#include <openssl/ssl.h>
#include "system_exception.hpp"
#include "log.hpp"

namespace Poseidon {

namespace {
	::pthread_once_t g_ssl_once = PTHREAD_ONCE_INIT;

	void init_ssl(){
		LOG_POSEIDON_INFO("Initializing SSL library...");

		::OpenSSL_add_all_algorithms();
		::SSL_library_init();

		std::atexit(&::EVP_cleanup);
	}

	UniqueSslCtx create_server_ssl_ctx(const char *cert, const char *private_key){
		const int err = ::pthread_once(&g_ssl_once, &init_ssl);
		if(err != 0){
			LOG_POSEIDON_FATAL("::pthread_once() failed with error code ", err);
			std::abort();
		}

		UniqueSslCtx ret;
		if(!ret.reset(::SSL_CTX_new(::SSLv23_server_method()))){
			LOG_POSEIDON_ERROR("Could not create server SSL context");
			DEBUG_THROW(SystemException, ENOMEM);
		}
		::SSL_CTX_set_verify(ret.get(), SSL_VERIFY_PEER | SSL_VERIFY_CLIENT_ONCE, NULLPTR);

		LOG_POSEIDON_INFO("Loading server certificate: ", cert);
		if(::SSL_CTX_use_certificate_file(ret.get(), cert, SSL_FILETYPE_PEM) != 1){
			DEBUG_THROW(Exception, sslit("::SSL_CTX_use_certificate_file() failed"));
		}

		LOG_POSEIDON_INFO("Loading server private key: ", private_key);
		if(::SSL_CTX_use_PrivateKey_file(ret.get(), private_key, SSL_FILETYPE_PEM) != 1){
			DEBUG_THROW(Exception, sslit("::SSL_CTX_use_PrivateKey_file() failed"));
		}

		LOG_POSEIDON_INFO("Verifying private key...");
		if(::SSL_CTX_check_private_key(ret.get()) != 1){
			DEBUG_THROW(Exception, sslit("::SSL_CTX_check_private_key() failed"));
		}

		return ret;
	}
	UniqueSslCtx create_client_ssl_ctx(){
		const int err = ::pthread_once(&g_ssl_once, &init_ssl);
		if(err != 0){
			LOG_POSEIDON_FATAL("::pthread_once() failed with error code ", err);
			std::abort();
		}

		UniqueSslCtx ret;
		if(!ret.reset(::SSL_CTX_new(::SSLv23_client_method()))){
			LOG_POSEIDON_ERROR("Could not create client SSL context");
			DEBUG_THROW(SystemException, ENOMEM);
		}
		::SSL_CTX_set_verify(ret.get(), SSL_VERIFY_NONE, NULLPTR);

		return ret;
	}
}

// SslFactoryBase
SslFactoryBase::SslFactoryBase(UniqueSslCtx ssl_ctx)
	: m_ssl_ctx(STD_MOVE(ssl_ctx))
{
}
SslFactoryBase::~SslFactoryBase(){
}

UniqueSsl SslFactoryBase::create_ssl() const {
	return UniqueSsl(::SSL_new(m_ssl_ctx.get()));
}

// ServerSslFactory
ServerSslFactory::ServerSslFactory(const char *cert, const char *private_key)
	: SslFactoryBase(create_server_ssl_ctx(cert, private_key))
{
}
ServerSslFactory::~ServerSslFactory(){
}

// ClientSslFactory
ClientSslFactory::ClientSslFactory()
	: SslFactoryBase(create_client_ssl_ctx())
{
}
ClientSslFactory::~ClientSslFactory(){
}

}
