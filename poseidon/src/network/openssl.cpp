// This file is part of Poseidon.
// Copyleft 2020, LH_Mouse. All wrongs reserved.

#include "../precompiled.hpp"
#include "openssl.hpp"
#include "../core/config_file.hpp"
#include "../static/main_config.hpp"
#include "../utilities.hpp"
#include <openssl/err.h>

namespace poseidon {
namespace {

unique_SSL_CTX
do_create_server_ssl_ctx(const char* cert, const char* pkey)
  {
    unique_SSL_CTX ctx(::SSL_CTX_new(::TLS_server_method()));
    if(!ctx)
      POSEIDON_SSL_THROW("could not create server SSL context\n"
                         "[`SSL_CTX_new()` failed]");

    POSEIDON_LOG_INFO("Loading SSL certificate: $1", cert);
    if(::SSL_CTX_use_certificate_chain_file(ctx, cert) != 1)
      POSEIDON_SSL_THROW("could not load certificate '$1'\n"
                         "[`SSL_CTX_use_certificate_chain_file()` failed]",
                         cert);

    POSEIDON_LOG_INFO("Loading SSL private key: $1", pkey);
    if(::SSL_CTX_use_PrivateKey_file(ctx, pkey, SSL_FILETYPE_PEM) != 1)
      POSEIDON_SSL_THROW("could not load private key '$1'\n"
                         "[`SSL_CTX_use_PrivateKey_file()` failed]",
                         pkey);

    if(::SSL_CTX_check_private_key(ctx) != 1)
      POSEIDON_SSL_THROW("SSL key pair check failure\n"
                         "[`SSL_CTX_check_private_key()` failed]",
                         cert, pkey);

    // Set the session ID.
    // This is carried over processes, so we use a hard-coded string in the executable.
    static constexpr unsigned char session_id[] = { __DATE__ __TIME__ };
    static_assert(sizeof(session_id) <= SSL_MAX_SSL_SESSION_ID_LENGTH);
    if(::SSL_CTX_set_session_id_context(ctx, session_id, sizeof(session_id)) != 1)
      POSEIDON_SSL_THROW("could not set SSL session id context\n"
                         "[`SSL_CTX_set_session_id_context()` failed]");

    ::SSL_CTX_set_verify(ctx, SSL_VERIFY_PEER, nullptr);
    return ctx;
  }

unique_SSL_CTX
do_create_default_server_ssl_ctx()
  {
    auto file = Main_Config::copy();
    auto qcert = file.get_string_opt({"network","tls","default_certificate"});
    auto kpkey = file.get_string_opt({"network","tls","default_private_key"});

    if(!qcert)
      POSEIDON_THROW("no default `network.tls.default_certificate` available");

    if(!kpkey)
      POSEIDON_THROW("no default `network.tls.default_private_key` available");

    return do_create_server_ssl_ctx(qcert->safe_c_str(), kpkey->safe_c_str());
  }

unique_SSL_CTX
do_create_default_client_ssl_ctx()
  {
    auto file = Main_Config::copy();
    auto ktpath = file.get_string_opt({"network","tls","trusted_ca_path"});

    unique_SSL_CTX ctx(::SSL_CTX_new(::TLS_client_method()));
    if(!ctx)
      POSEIDON_SSL_THROW("could not create client SSL context\n"
                         "[`SSL_CTX_new()` failed]");

    if(!ktpath) {
      POSEIDON_LOG_WARN("Note: CA certificate validation has been disabled."
                        " This configuration is not suitable for production use.\n"
                        "Set `network.tls.trusted_ca_path` to enable.");
      ::SSL_CTX_set_verify(ctx, SSL_VERIFY_NONE, nullptr);
      return ctx;
    }

    POSEIDON_LOG_INFO("Setting SSL CA certificate path: $1", *ktpath);
    if(::SSL_CTX_load_verify_locations(ctx, nullptr, ktpath->safe_c_str()) != 1)
      POSEIDON_SSL_THROW("could not set CA certificate path to '$1'\n"
                         "[`SSL_CTX_set_default_verify_paths()` failed]",
                         *ktpath);

    ::SSL_CTX_set_verify(ctx, SSL_VERIFY_PEER, nullptr);
    return ctx;
  }

unique_SSL_CTX
do_copy_default_ssl_ctx(::SSL_CTX* ctx)
  {
    int res = ::SSL_CTX_up_ref(ctx);
    ROCKET_ASSERT(res == 1);
    return unique_SSL_CTX(ctx);
  }

}  // namespace

size_t
dump_ssl_errors()
noexcept
  {
    char sbuf[1024];
    size_t index = 0;

    while(unsigned long err = ::ERR_get_error()) {
      ::ERR_error_string_n(err, sbuf, sizeof(sbuf));
      POSEIDON_LOG_ERROR("OpenSSL error: [$1] $2", index, sbuf);
      ++index;
    }
    return index;
  }

unique_SSL_CTX
create_server_ssl_ctx(const char* cert_opt, const char* pkey_opt)
  {
    if(cert_opt && cert_opt)
      return do_create_server_ssl_ctx(cert_opt, pkey_opt);

    if(cert_opt || cert_opt)
      POSEIDON_THROW("certificate and private key must be both specified or both absent");

    // Cache the default context.
    static const auto s_default_ctx = do_create_default_server_ssl_ctx();
    ROCKET_ASSERT(s_default_ctx);
    return do_copy_default_ssl_ctx(s_default_ctx);
  }

::SSL_CTX*
get_client_ssl_ctx()
  {
    // Cache the default context.
    static const auto s_default_ctx = do_create_default_client_ssl_ctx();
    ROCKET_ASSERT(s_default_ctx);
    return do_copy_default_ssl_ctx(s_default_ctx);
  }

unique_SSL
create_ssl(::SSL_CTX* ctx, int fd)
  {
    // Create SSL structure and assign `fd` to it.
    unique_SSL ssl(::SSL_new(ctx));
    if(!ssl)
      POSEIDON_SSL_THROW("could not create SSL structure\n"
                         "[`SSL_new()` failed]");

    if(::SSL_set_fd(ssl, fd) != 1)
      POSEIDON_SSL_THROW("could not set OpenSSL file descriptor\n"
                         "[`SSL_set_fd()` failed]");

    // This can be overwritten if `async_connect()` is called later.
    ::SSL_set_accept_state(ssl);
    return ssl;
  }

}  // namespace poseidon
