// This file is part of Poseidon.
// Copyleft 2020, LH_Mouse. All wrongs reserved.

#include "../precompiled.ipp"
#include "openssl_stream.hpp"
#include "openssl_context.hpp"
#include "abstract_socket.hpp"
#include "../utils.hpp"

namespace poseidon {

OpenSSL_Stream::
OpenSSL_Stream(const OpenSSL_Context& ctx, const Abstract_Socket& sock)
  {
    // Note that it is safe to share `SSL_CTX` objects amongst threads.
    // However `SSL_new()` does not take a `const SSL_CTX*` argument, probably
    // because it needs to bump up the reference counter of the context.
    this->m_ssl.reset(::SSL_new(const_cast<::SSL_CTX*>(ctx.get_ssl_ctx())));
    if(!this->m_ssl)
      POSEIDON_SSL_THROW(
          "could not create SSL structure\n"
          "[`SSL_new()` failed]");

    if(::SSL_set_fd(this->m_ssl, sock.get_fd()) != 1)
      POSEIDON_SSL_THROW(
          "could not set file descriptor\n"
          "[`SSL_set_fd()` failed]");
  }

OpenSSL_Stream::
~OpenSSL_Stream()
  {
  }

}  // namespace poseidon
