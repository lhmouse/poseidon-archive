// This file is part of Poseidon.
// Copyleft 2020, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_DETAILS_OPENSSL_COMMON_HPP_
#define POSEIDON_DETAILS_OPENSSL_COMMON_HPP_

#include "../fwd.hpp"
#include <openssl/ssl.h>

namespace poseidon {
namespace details_openssl_common {

struct CTX_deleter
  {
    void
    operator()(::SSL_CTX* ctx) noexcept
      { ::SSL_CTX_free(ctx);  }
  };

struct SSL_deleter
  {
    void
    operator()(::SSL* ssl) noexcept
      { ::SSL_free(ssl);  }
  };

using unique_CTX = uptr<::SSL_CTX, CTX_deleter>;
using unique_SSL = uptr<::SSL, SSL_deleter>;

// Prints all OpenSSL errors and clears the error queue.
void
log_openssl_errors() noexcept;

#define POSEIDON_SSL_THROW(...)   \
    (::poseidon::details_openssl_common::log_openssl_errors(),  \
       POSEIDON_THROW(__VA_ARGS__))

}  // namespace details_openssl_common
}  // namespace poseidon

#endif
