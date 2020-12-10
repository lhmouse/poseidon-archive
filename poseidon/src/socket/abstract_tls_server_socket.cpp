// This file is part of Poseidon.
// Copyleft 2020, LH_Mouse. All wrongs reserved.

#include "../precompiled.hpp"
#include "abstract_tls_server_socket.hpp"
#include "abstract_tls_socket.hpp"
#include "../static/main_config.hpp"
#include "../core/config_file.hpp"
#include "../utils.hpp"

namespace poseidon {
namespace {

void
do_load_cert_and_pkey(::SSL_CTX* ctx, const char* cert, const char* pkey)
  {
    POSEIDON_LOG_INFO("Loading SSL certificate '$1'...", cert);
    if(::SSL_CTX_use_certificate_chain_file(ctx, cert) != 1)
      POSEIDON_SSL_THROW("Could not load certificate '$1'\n"
                         "[`SSL_CTX_use_certificate_chain_file()` failed]",
                         cert);

    POSEIDON_LOG_INFO("Loading SSL private key '$1'...", pkey);
    if(::SSL_CTX_use_PrivateKey_file(ctx, pkey, SSL_FILETYPE_PEM) != 1)
      POSEIDON_SSL_THROW("Could not load private key '$1'\n"
                         "[`SSL_CTX_use_PrivateKey_file()` failed]",
                         pkey);

    if(::SSL_CTX_check_private_key(ctx) != 1)
      POSEIDON_SSL_THROW("SSL key/certificate pair check failure\n"
                         "[`SSL_CTX_check_private_key()` failed]",
                         cert, pkey);

    ::SSL_CTX_set_verify(ctx, SSL_VERIFY_PEER | SSL_VERIFY_CLIENT_ONCE, nullptr);
  }

}  // namespace

Abstract_TLS_Server_Socket::
Abstract_TLS_Server_Socket(const Socket_Address& addr)
  : Abstract_Accept_Socket(addr.family())
  {
    // Load the default key/certificate pair in 'main.conf'.
    const auto file = Main_Config::copy();

    auto cert = file.get_string_opt({"network","tls","default_certificate"});
    if(!cert)
      POSEIDON_THROW("`network.tls.default_certificate` not configured");

    auto pkey = file.get_string_opt({"network","tls","default_private_key"});
    if(!pkey)
      POSEIDON_THROW("`network.tls.default_private_key` not configured");

    do_load_cert_and_pkey(this->open_ssl_ctx(), cert->safe_c_str(),
                          pkey->safe_c_str());
    this->do_socket_listen(addr);
  }

Abstract_TLS_Server_Socket::
Abstract_TLS_Server_Socket(const Socket_Address& addr, const char* cert,
                           const char* pkey)
  : Abstract_Accept_Socket(addr.family())
  {
    // Load the user-specified key/certificate pair.
    if(!cert)
      POSEIDON_THROW("TLS server requires a certificate");

    if(!pkey)
      POSEIDON_THROW("TLS server requires a private key");

    do_load_cert_and_pkey(this->open_ssl_ctx(), cert, pkey);
    this->do_socket_listen(addr);
  }

Abstract_TLS_Server_Socket::
~Abstract_TLS_Server_Socket()
  {
  }

uptr<Abstract_Socket>
Abstract_TLS_Server_Socket::
do_socket_on_accept(unique_FD&& fd)
  {
    return this->do_socket_on_accept_tls(::std::move(fd));
  }

void
Abstract_TLS_Server_Socket::
do_socket_on_register(rcptr<Abstract_Socket>&& sock)
  {
    return this->do_socket_on_register_tls(
        ::rocket::static_pointer_cast<Abstract_TLS_Socket>(
                    ::std::move(sock)));
  }

}  // namespace poseidon
