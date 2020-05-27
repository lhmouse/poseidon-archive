// This file is part of Poseidon.
// Copyleft 2020, LH_Mouse. All wrongs reserved.

#include "precompiled.hpp"
#include "xutilities.hpp"
#include <openssl/err.h>

namespace poseidon {

size_t
dump_ssl_errors()
noexcept
  {
    char sbuf[1024];
    size_t index = 0;

    while(unsigned long err = ::ERR_get_error()) {
      ::ERR_error_string_n(err, sbuf, sizeof(sbuf));
      POSEIDON_LOG_ERROR("OpenSSL error: [$1] $2", index, err);
      ++index;
    }
    return index;
  }

}  // namespace poseidon
