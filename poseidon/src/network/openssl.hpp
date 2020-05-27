// This file is part of Poseidon.
// Copyleft 2020, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_NETWORK_OPENSSL_HPP_
#define POSEIDON_NETWORK_OPENSSL_HPP_

#include "../fwd.hpp"
#include <openssl/ssl.h>

namespace poseidon {

struct SSL_deleter
  {
    void
    operator()(::SSL* ssl)
    noexcept
      { ::SSL_free(ssl);  }
  };

using unique_SSL = uptr<::SSL, SSL_deleter>;

struct SSL_CTX_deleter
  {
    void
    operator()(::SSL_CTX* ctx)
    noexcept
      { ::SSL_CTX_free(ctx);  }
  };

using unique_SSL_CTX = uptr<::SSL_CTX, SSL_CTX_deleter>;

extern
void
dump_ssl_errors()
noexcept;

}  // namespace poseidon

#endif
