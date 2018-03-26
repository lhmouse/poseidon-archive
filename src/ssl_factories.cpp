// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2018, LH_Mouse. All wrongs reserved.

#include "precompiled.hpp"
#include "ssl_factories.hpp"
#include "ssl_filter.hpp"
#include "exception.hpp"
#include "log.hpp"
#include "profiler.hpp"
#include "singletons/main_config.hpp"
#include <openssl/ssl.h>

namespace Poseidon {

#if OPENSSL_VERSION_NUMBER < 0x10100000L

namespace {
	void init_openssl(){
		::OpenSSL_add_all_algorithms();
		::OpenSSL_add_all_ciphers();
		::OpenSSL_add_all_digests();
		::SSL_load_error_strings();
		::SSL_library_init();

		std::atexit(&::EVP_cleanup);
	}

	::pthread_once_t g_ssl_once = PTHREAD_ONCE_INIT;
}

void init_ssl_once(){
	DEBUG_THROW_ASSERT(::pthread_once(&g_ssl_once, &init_openssl) == 0);
}

#else

void init_ssl_once(){
	// Since OpenSSL 1.1.0 no explicit initialization or cleanup is required.
}

#endif

namespace {
	Unique_ssl_ctx create_server_ssl_ctx(const char *certificate, const char *private_key){
		PROFILE_ME;

		init_ssl_once();

		Unique_ssl_ctx ssl_ctx;
		DEBUG_THROW_UNLESS(ssl_ctx.reset(::SSL_CTX_new(::SSLv23_server_method())), Exception, sslit("::SSLv23_server_method() failed"));
		::SSL_CTX_set_options(ssl_ctx.get(), SSL_OP_NO_SSLv2);
		::SSL_CTX_set_options(ssl_ctx.get(), SSL_OP_NO_SSLv3);
		if(certificate && *certificate){
			LOG_POSEIDON_INFO("Loading server certificate: ", certificate);
			DEBUG_THROW_UNLESS(::SSL_CTX_use_certificate_chain_file(ssl_ctx.get(), certificate) == 1, Exception, sslit("::SSL_CTX_use_certificate_file() failed"));
			LOG_POSEIDON_INFO("Loading server private key: ", private_key);
			DEBUG_THROW_UNLESS(::SSL_CTX_use_PrivateKey_file(ssl_ctx.get(), private_key, SSL_FILETYPE_PEM) == 1, Exception, sslit("::SSL_CTX_use_PrivateKey_file() failed"));
			LOG_POSEIDON_INFO("Verifying private key...");
			DEBUG_THROW_UNLESS(::SSL_CTX_check_private_key(ssl_ctx.get()) == 1, Exception, sslit("::SSL_CTX_check_private_key() failed"));
			LOG_POSEIDON_INFO("Setting session ID context...");
			static CONSTEXPR const unsigned char ssl_session_id[SSL_MAX_SSL_SESSION_ID_LENGTH] = { __DATE__ __TIME__ };
			DEBUG_THROW_UNLESS(::SSL_CTX_set_session_id_context(ssl_ctx.get(), ssl_session_id, sizeof(ssl_session_id)) == 1, Exception, sslit("::SSL_CTX_set_session_id_context() failed"));
			::SSL_CTX_set_verify(ssl_ctx.get(), SSL_VERIFY_PEER, NULLPTR);
		} else {
			::SSL_CTX_set_verify(ssl_ctx.get(), SSL_VERIFY_NONE, NULLPTR);
		}
		return ssl_ctx;
	}

	Unique_ssl_ctx create_client_ssl_ctx(bool verify_peer){
		PROFILE_ME;

		init_ssl_once();

		Unique_ssl_ctx ssl_ctx;
		DEBUG_THROW_UNLESS(ssl_ctx.reset(::SSL_CTX_new(::SSLv23_client_method())), Exception, sslit("::SSLv23_client_method() failed"));
		::SSL_CTX_set_options(ssl_ctx.get(), SSL_OP_NO_SSLv2);
		::SSL_CTX_set_options(ssl_ctx.get(), SSL_OP_NO_SSLv3);
		if(verify_peer){
			const AUTO(ssl_cert_directory, Main_config::get<std::string>("ssl_cert_directory", "/etc/ssl/certs"));
			LOG_POSEIDON_INFO("Loading trusted CA certificates: ", ssl_cert_directory);
			DEBUG_THROW_UNLESS(::SSL_CTX_load_verify_locations(ssl_ctx.get(), NULLPTR, ssl_cert_directory.c_str()) == 1, Exception, sslit("::SSL_CTX_load_verify_locations() failed"));
			::SSL_CTX_set_verify(ssl_ctx.get(), SSL_VERIFY_PEER, NULLPTR);
		} else {
			::SSL_CTX_set_verify(ssl_ctx.get(), SSL_VERIFY_NONE, NULLPTR);
		}
		return ssl_ctx;
	}
}

Ssl_server_factory::Ssl_server_factory(const char *certificate, const char *private_key)
	: m_ssl_ctx(create_server_ssl_ctx(certificate, private_key))
{
	//
}
Ssl_server_factory::~Ssl_server_factory(){
	//
}

void Ssl_server_factory::create_ssl_filter(boost::scoped_ptr<Ssl_filter> &ssl_filter, int fd){
	Unique_ssl ssl;
	DEBUG_THROW_UNLESS(ssl.reset(::SSL_new(m_ssl_ctx.get())), Exception, sslit("::SSL_new() failed"));
	ssl_filter.reset(new Ssl_filter(STD_MOVE(ssl), Ssl_filter::to_accept, fd));
}

Ssl_client_factory::Ssl_client_factory(bool verify_peer)
	: m_ssl_ctx(create_client_ssl_ctx(verify_peer))
{
	//
}
Ssl_client_factory::~Ssl_client_factory(){
	//
}

void Ssl_client_factory::create_ssl_filter(boost::scoped_ptr<Ssl_filter> &ssl_filter, int fd){
	Unique_ssl ssl;
	DEBUG_THROW_UNLESS(ssl.reset(::SSL_new(m_ssl_ctx.get())), Exception, sslit("::SSL_new() failed"));
	ssl_filter.reset(new Ssl_filter(STD_MOVE(ssl), Ssl_filter::to_connect, fd));
}

}
