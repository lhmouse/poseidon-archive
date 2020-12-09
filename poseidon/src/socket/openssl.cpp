// This file is part of Poseidon.
// Copyleft 2020, LH_Mouse. All wrongs reserved.

#include "../precompiled.hpp"
#include "openssl.hpp"
#include "enums.hpp"
#include "../core/config_file.hpp"
#include "../static/main_config.hpp"
#include "../utils.hpp"
#include <openssl/err.h>

namespace poseidon {
namespace {

unique_SSL_CTX
do_create_server_ssl_ctx(const char* cert, const char* pkey)
  {
    unique_SSL_CTX ctx(::SSL_CTX_new(::TLS_server_method()));
    if(!ctx)
      POSEIDON_THROW("Could not create server SSL context\n"
                         "[`SSL_CTX_new()` failed]");

    POSEIDON_LOG_INFO("Loading SSL certificate: $1", cert);
    if(::SSL_CTX_use_certificate_chain_file(ctx, cert) != 1)
      POSEIDON_THROW("Could not load certificate '$1'\n"
                     "[`SSL_CTX_use_certificate_chain_file()` failed]",
                     cert);

    POSEIDON_LOG_INFO("Loading SSL private key: $1", pkey);
    if(::SSL_CTX_use_PrivateKey_file(ctx, pkey, SSL_FILETYPE_PEM) != 1)
      POSEIDON_THROW("Could not load private key '$1'\n"
                     "[`SSL_CTX_use_PrivateKey_file()` failed]",
                     pkey);

    if(::SSL_CTX_check_private_key(ctx) != 1)
      POSEIDON_THROW("SSL key pair check failure\n"
                     "[`SSL_CTX_check_private_key()` failed]",
                     cert, pkey);

    // Set the session ID.
    // This is carried over processes, so we use a hard-coded string in the executable.
    static constexpr unsigned char session_id[] = { __DATE__ __TIME__ };
    static_assert(sizeof(session_id) <= SSL_MAX_SSL_SESSION_ID_LENGTH);
    if(::SSL_CTX_set_session_id_context(ctx, session_id, sizeof(session_id)) != 1)
      POSEIDON_THROW("Could not set SSL session id context\n"
                     "[`SSL_CTX_set_session_id_context()` failed]");

    ::SSL_CTX_set_verify(ctx, SSL_VERIFY_PEER, nullptr);
    return ctx;
  }

unique_SSL_CTX
do_create_default_server_ssl_ctx()
  {
    auto file = Main_Config::copy();

    auto qcert = file.get_string_opt({"network","tls","default_certificate"});
    if(!qcert)
      POSEIDON_THROW("No default `network.tls.default_certificate` available");

    auto kpkey = file.get_string_opt({"network","tls","default_private_key"});
    if(!kpkey)
      POSEIDON_THROW("No default `network.tls.default_private_key` available");

    return do_create_server_ssl_ctx(qcert->safe_c_str(), kpkey->safe_c_str());
  }

unique_SSL_CTX
do_create_default_client_ssl_ctx()
  {
    auto file = Main_Config::copy();

    unique_SSL_CTX ctx(::SSL_CTX_new(::TLS_client_method()));
    if(!ctx)
      POSEIDON_THROW("Could not create client SSL context\n"
                     "[`SSL_CTX_new()` failed]");

    auto ktpath = file.get_string_opt({"network","tls","trusted_ca_path"});
    if(!ktpath) {
      POSEIDON_LOG_WARN("Note: CA certificate validation has been disabled."
                        " This configuration is not suitable for production use.\n"
                        "      Set `network.tls.trusted_ca_path` to enable.");

      ::SSL_CTX_set_verify(ctx, SSL_VERIFY_NONE, nullptr);
      return ctx;
    }
    POSEIDON_LOG_INFO("Setting SSL CA certificate path: $1", *ktpath);

    if(::SSL_CTX_load_verify_locations(ctx, nullptr, ktpath->safe_c_str()) != 1)
      POSEIDON_THROW("Could not set CA certificate path to '$1'\n"
                     "[`SSL_CTX_set_default_verify_paths()` failed]",
                     *ktpath);

    ::SSL_CTX_set_verify(ctx, SSL_VERIFY_PEER, nullptr);
    return ctx;
  }

}  // namespace

unique_SSL_CTX
create_server_ssl_ctx(const char* cert_opt, const char* pkey_opt)
  {
    if(cert_opt && cert_opt)
      return do_create_server_ssl_ctx(cert_opt, pkey_opt);

    if(cert_opt || cert_opt)
      POSEIDON_THROW("Certificate and private key must be both "
                     "specified or both absent");

    // Create the default context.
    static const auto s_default_ctx = do_create_default_server_ssl_ctx();
    auto ctx = s_default_ctx.get();
    ROCKET_ASSERT(ctx);

    // Bump up its refrence count and return an owning pointer to it.
    int res = ::SSL_CTX_up_ref(ctx);
    ROCKET_ASSERT(res == 1);
    return unique_SSL_CTX(ctx);
  }

::SSL_CTX*
get_client_ssl_ctx()
  {
    // Create the default context.
    static const auto s_default_ctx = do_create_default_client_ssl_ctx();
    auto ctx = s_default_ctx.get();
    ROCKET_ASSERT(ctx);

    // Return a non-owning pointer to it.
    return ctx;
  }

unique_SSL
create_ssl(::SSL_CTX* ctx, int fd)
  {
    unique_SSL ssl(::SSL_new(ctx));
    if(!ssl)
      POSEIDON_THROW("Could not create SSL structure\n"
                     "[`SSL_new()` failed]");

    if(::SSL_set_fd(ssl, fd) != 1)
      POSEIDON_THROW("Could not set OpenSSL file descriptor\n"
                     "[`SSL_set_fd()` failed]");

    // Return the I/O object.
    return ssl;
  }

}  // namespace poseidon
