// This file is part of Poseidon.
// Copyleft 2020, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_SOCKET_OPENSSL_STREAM_HPP_
#define POSEIDON_SOCKET_OPENSSL_STREAM_HPP_

#include "../fwd.hpp"
#include "../details/openssl_common.hpp"

namespace poseidon {

class OpenSSL_Stream
  : public ::asteria::Rcfwd<OpenSSL_Stream>
  {
  private:
    details_openssl_common::unique_SSL m_ssl;

  public:
    // Creates a new SSL structure.
    explicit
    OpenSSL_Stream(const OpenSSL_Context& ctx, const Abstract_Socket& sock);

  public:
    ASTERIA_NONCOPYABLE_DESTRUCTOR(OpenSSL_Stream);

    // Get the SSL structure for other OpenSSL APIs.
    const ::SSL*
    get_ssl()
      const noexcept
      { return this->m_ssl;  }

    ::SSL*
    open_ssl()
      noexcept
      { return this->m_ssl;  }
  };

}  // namespace poseidon

#endif
