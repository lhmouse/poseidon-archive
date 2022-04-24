// This file is part of Poseidon.
// Copyleft 2020, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_SOCKET_OPENSSL_CONTEXT_HPP_
#define POSEIDON_SOCKET_OPENSSL_CONTEXT_HPP_

#include "../fwd.hpp"
#include "../details/openssl_common.hpp"

namespace poseidon {

class OpenSSL_Context
  : public ::asteria::Rcfwd<OpenSSL_Context>
  {
  public:
    // Returns a static context that has been created with `SSL_VERIFY_PEER`.
    // This is the precommended context for servers and clients.
    ROCKET_CONST static
    const OpenSSL_Context&
    static_verify_peer();

    // Returns a static context that has been created with `SSL_VERIFY_NONE`.
    // This is the precommended context for servers without certificates, and
    // for clients that don't have to validate server certificates. This may
    // be useful for private communication.
    ROCKET_CONST static
    const OpenSSL_Context&
    static_verify_none();

  private:
    details_openssl_common::unique_CTX m_ctx;

  public:
    // Creates a new SSL context.
    explicit
    OpenSSL_Context();

  public:
    ASTERIA_NONCOPYABLE_DESTRUCTOR(OpenSSL_Context);

    // Get the SSL context for other OpenSSL APIs.
    const ::SSL_CTX*
    get_ssl_ctx() const noexcept
      { return this->m_ctx;  }

    ::SSL_CTX*
    open_ssl_ctx() noexcept
      { return this->m_ctx;  }
  };

}  // namespace poseidon

#endif
