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
	::pthread_once_t g_sslOnce = PTHREAD_ONCE_INIT;

	void initSsl(){
		LOG_POSEIDON_INFO("Initializing SSL library...");

		::OpenSSL_add_all_algorithms();
		::SSL_library_init();

		std::atexit(&::EVP_cleanup);
	}

	UniqueSslCtx createServerSslCtx(const char *cert, const char *privateKey){
		const int err = ::pthread_once(&g_sslOnce, &initSsl);
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
			DEBUG_THROW(Exception, SSLIT("::SSL_CTX_use_certificate_file() failed"));
		}

		LOG_POSEIDON_INFO("Loading server private key: ", privateKey);
		if(::SSL_CTX_use_PrivateKey_file(ret.get(), privateKey, SSL_FILETYPE_PEM) != 1){
			DEBUG_THROW(Exception, SSLIT("::SSL_CTX_use_PrivateKey_file() failed"));
		}

		LOG_POSEIDON_INFO("Verifying private key...");
		if(::SSL_CTX_check_private_key(ret.get()) != 1){
			DEBUG_THROW(Exception, SSLIT("::SSL_CTX_check_private_key() failed"));
		}

		return ret;
	}
	UniqueSslCtx createClientSslCtx(){
		const int err = ::pthread_once(&g_sslOnce, &initSsl);
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
SslFactoryBase::SslFactoryBase(UniqueSslCtx sslCtx)
	: m_sslCtx(STD_MOVE(sslCtx))
{
}
SslFactoryBase::~SslFactoryBase(){
}

UniqueSsl SslFactoryBase::createSsl() const {
	return UniqueSsl(::SSL_new(m_sslCtx.get()));
}

// ServerSslFactory
ServerSslFactory::ServerSslFactory(const char *cert, const char *privateKey)
	: SslFactoryBase(createServerSslCtx(cert, privateKey))
{
}
ServerSslFactory::~ServerSslFactory(){
}

// ClientSslFactory
ClientSslFactory::ClientSslFactory()
	: SslFactoryBase(createClientSslCtx())
{
}
ClientSslFactory::~ClientSslFactory(){
}

}
