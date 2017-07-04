// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2017, LH_Mouse. All wrongs reserved.

#include "precompiled.hpp"
#include "ssl_factories.hpp"
#include <openssl/ssl.h>
#include "system_exception.hpp"
#include "log.hpp"
#include "singletons/main_config.hpp"

namespace Poseidon {

namespace {
	::pthread_once_t g_ssl_once = PTHREAD_ONCE_INIT;

	void init_ssl(){
		LOG_POSEIDON_INFO("Initializing OpenSSL...");

		::OpenSSL_add_all_algorithms();
		::SSL_load_error_strings();
		::SSL_library_init();

		std::atexit(&::EVP_cleanup);
	}
}

SslFactoryBase::SslFactoryBase(){
	const int err_code = ::pthread_once(&g_ssl_once, &init_ssl);
	if(err_code != 0){
		LOG_POSEIDON_FATAL("::pthread_once() failed with error code ", err_code);
		std::abort();
	}
}
SslFactoryBase::~SslFactoryBase(){ }

void SslFactoryBase::create_ssl(UniqueSsl &ssl) const {
	ssl.reset(::SSL_new(m_ctx.get()));
}

ServerSslFactory::ServerSslFactory(const char *cert, const char *private_key)
	: SslFactoryBase()
{
	if(!m_ctx.reset(::SSL_CTX_new(::SSLv23_server_method()))){
		LOG_POSEIDON_ERROR("Could not create server SSL context");
		DEBUG_THROW(SystemException, ENOMEM);
	}
	::SSL_CTX_set_options(m_ctx.get(), SSL_OP_NO_SSLv2);
	::SSL_CTX_set_options(m_ctx.get(), SSL_OP_NO_SSLv3);
	::SSL_CTX_set_verify(m_ctx.get(), SSL_VERIFY_PEER, NULLPTR);

	LOG_POSEIDON_INFO("Loading server certificate: ", cert);
	if(::SSL_CTX_use_certificate_chain_file(m_ctx.get(), cert) != 1){
		DEBUG_THROW(Exception, sslit("::SSL_CTX_use_certificate_file() failed"));
	}

	LOG_POSEIDON_INFO("Loading server private key: ", private_key);
	if(::SSL_CTX_use_PrivateKey_file(m_ctx.get(), private_key, SSL_FILETYPE_PEM) != 1){
		DEBUG_THROW(Exception, sslit("::SSL_CTX_use_PrivateKey_file() failed"));
	}

	LOG_POSEIDON_INFO("Verifying private key...");
	if(::SSL_CTX_check_private_key(m_ctx.get()) != 1){
		DEBUG_THROW(Exception, sslit("::SSL_CTX_check_private_key() failed"));
	}

	static CONSTEXPR const unsigned char s_session_id_context[SSL_MAX_SSL_SESSION_ID_LENGTH] =
		{ 0x15, 0x74, 0xFA, 0x87, 0x85, 0x70, 0x00, 0x08, 0xD2, 0xFD, 0x47, 0xC3, 0x84, 0xE3, 0x19, 0xDD };
	if(::SSL_CTX_set_session_id_context(m_ctx.get(), s_session_id_context, sizeof(s_session_id_context)) != 1){
		DEBUG_THROW(Exception, sslit("::SSL_CTX_set_session_id_context() failed"));
	}
}
ServerSslFactory::~ServerSslFactory(){ }

ClientSslFactory::ClientSslFactory(bool verify_peer)
	: SslFactoryBase()
{
	if(!m_ctx.reset(::SSL_CTX_new(::SSLv23_client_method()))){
		LOG_POSEIDON_ERROR("Could not create client SSL context");
		DEBUG_THROW(SystemException, ENOMEM);
	}
	::SSL_CTX_set_options(m_ctx.get(), SSL_OP_NO_SSLv2);
	::SSL_CTX_set_options(m_ctx.get(), SSL_OP_NO_SSLv3);
	if(verify_peer){
		const AUTO(ssl_cert_directory, MainConfig::get<std::string>("ssl_cert_directory", "/etc/ssl/certs"));
		if(::SSL_CTX_load_verify_locations(m_ctx.get(), NULLPTR, ssl_cert_directory.c_str()) != 1){
			DEBUG_THROW(Exception, sslit("::SSL_CTX_load_verify_locations() failed"));
		}
		::SSL_CTX_set_verify(m_ctx.get(), SSL_VERIFY_PEER, NULLPTR);
	} else {
		::SSL_CTX_set_verify(m_ctx.get(), SSL_VERIFY_NONE, NULLPTR);
	}
}
ClientSslFactory::~ClientSslFactory(){ }

}
