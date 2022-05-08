// This file is part of Poseidon.
// Copyleft 2020, LH_Mouse. All wrongs reserved.

#include "../precompiled.ipp"
#include "openssl_common.hpp"
#include "../utils.hpp"
#include <openssl/err.h>

namespace poseidon {
namespace details_openssl_common {

void
log_openssl_errors() noexcept
  {
    unsigned long err;
    long index = -1;
    char sbuf[512];

    while((err = ::ERR_get_error()) != 0)
      ::ERR_error_string_n(err, sbuf, sizeof(sbuf)),
        POSEIDON_LOG_WARN("OpenSSL error: [$1] $2", ++index, sbuf);
  }

}  // namespace details_openssl_common
}  // namespace poseidon
