// This file is part of Poseidon.
// Copyleft 2020, LH_Mouse. All wrongs reserved.

#include "../precompiled.hpp"
#include "openssl_io.hpp"
#include "../utilities.hpp"
#include <openssl/err.h>

namespace poseidon {
namespace {

ROCKET_NOINLINE
size_t
do_print_errors()
  {
    char sbuf[1024];
    size_t index = 0;

    while(auto err = ::ERR_get_error()) {
      ::ERR_error_string_n(err, sbuf, sizeof(sbuf));
      POSEIDON_LOG_ERROR("OpenSSL error: [$1] $2", index, err);
      ++index;
    }
    return index;
  }

int
do_errno_from(::SSL* ssl, int ret)
  {
    int err = ::SSL_get_error(ssl, ret);
    switch(err) {
      case SSL_ERROR_NONE:
      case SSL_ERROR_ZERO_RETURN:
        // normal closure
        err = 0;
        break;

      case SSL_ERROR_WANT_READ:
      case SSL_ERROR_WANT_WRITE:
      case SSL_ERROR_WANT_CONNECT:
      case SSL_ERROR_WANT_ACCEPT:
        // retry
        err = EAGAIN;
        break;

      case SSL_ERROR_SYSCALL:
        // forward `errno` verbatim
        err = errno;
        do_print_errors();
        break;

      default:
        // generic failure
        err = EPERM;
        do_print_errors();
        break;
    }
    return err;
  }

}  // namespace

OpenSSL_IO::
~OpenSSL_IO()
  {
  }

void
OpenSSL_IO::
set_method(Method method)
noexcept
  {
    if(method == method_connect)
      ::SSL_set_connect_state(this->m_ssl);

    if(method == method_accept)
      ::SSL_set_accept_state(this->m_ssl);
  }

ptrdiff_t
OpenSSL_IO::
read(void* data, size_t size)
  {
    // Try reading some data.
    // Note: According to OpenSSL documentation, unlike `::read()`,
    //       a return value of zero indates failure.
    int nread = ::SSL_read(this->m_ssl, data,
                           static_cast<int>(::rocket::min(size, UINT_MAX / 2)));
    if(nread > 0)
      return nread;  // success

    // Find out why.
    int err = do_errno_from(this->m_ssl, nread);
    if(err == 0)
      return 0;  // normal closure

    if(::rocket::is_any_of(err, { EINTR, EAGAIN, EWOULDBLOCK }))
      return -1;  // retry

    POSEIDON_THROW("OpenSSL read error\n"
                   "[`SSL_read()` returned `$1`: $2]",
                   nread, format_errno(err));
  }

ptrdiff_t
OpenSSL_IO::
write(const void* data, size_t size)
  {
    // Try writing some data.
    // Note: According to OpenSSL documentation, unlike `::write()`,
    //       a return value of zero indates failure.
    int nwritten = ::SSL_write(this->m_ssl, data,
                               static_cast<int>(::rocket::min(size, UINT_MAX / 2)));
    if(nwritten > 0)
      return nwritten;  // success

    // Find out why.
    int err = do_errno_from(this->m_ssl, nwritten);
    if(err == 0)
      return 0;  // ?????

    if(::rocket::is_any_of(err, { EINTR, EAGAIN, EWOULDBLOCK }))
      return -1;  // retry

    POSEIDON_THROW("OpenSSL write error\n"
                   "[`SSL_write()` returned `$1`: $2]",
                   nwritten, format_errno(err));
  }

bool
OpenSSL_IO::
shutdown()
  {
    // Perform normal closure.
    int result = ::SSL_shutdown(this->m_ssl);
    if(result == 1)
      return true;  // complete

    if(result == 0)
      return false;  // not complete

    // Find out why.
    int err = do_errno_from(this->m_ssl, result);
    if(err == 0)
      return true;  // ?????

    POSEIDON_THROW("OpenSSL shutdown error\n"
                   "[`SSL_shutdown()` returned `$1`: $2]",
                   result, format_errno(err));
  }

}  // namespace poseidon
