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

    while(unsigned long err = ::ERR_get_error()) {
      ::ERR_error_string_n(err, sbuf, sizeof(sbuf));
      POSEIDON_LOG_ERROR("OpenSSL error: [$1] $2", index, err);
      ++index;
    }
    return index;
  }

IO_Result
do_translate_ssl_error(const char* func, ::SSL* ssl, int res)
  {
    int err = ::SSL_get_error(ssl, res);
    switch(err) {
      case SSL_ERROR_NONE:
      case SSL_ERROR_ZERO_RETURN:
        // normal closure
        return io_result_eof;

      case SSL_ERROR_WANT_READ:
      case SSL_ERROR_WANT_WRITE:
      case SSL_ERROR_WANT_CONNECT:
      case SSL_ERROR_WANT_ACCEPT:
        // retry
        return io_result_again;

      case SSL_ERROR_SYSCALL:
        // dump and clear the thread-local error queue
        do_print_errors();

        // forward `errno` verbatim
        err = errno;
        if(err == EINTR)
          return io_result_intr;

        POSEIDON_THROW("OpenSSL reported an irrecoverable I/O error\n"
                       "[`$1()` returned `$2`; errno: $3]",
                       func, res, noadl::format_errno(errno));

      default:
        // dump and clear the thread-local error queue
        do_print_errors();

        // report generic failure
        POSEIDON_THROW("OpenSSL reported an irrecoverable error\n"
                       "[`$1()` returned `$2`]",
                       func, res);
    }
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

IO_Result
OpenSSL_IO::
read(void* data, size_t size)
  {
    // Try reading some data.
    // Note: According to OpenSSL documentation, unlike `::read()`, a return
    //       value of zero indates failure, rather than EOF.
    int res = ::SSL_read(this->m_ssl, data,
                         static_cast<int>(::rocket::min(size, UINT_MAX / 2)));
    if(res > 0)
      return static_cast<IO_Result>(res);  // success

    return do_translate_ssl_error("SSL_read", this->m_ssl, res);
  }

IO_Result
OpenSSL_IO::
write(const void* data, size_t size)
  {
    // Try writing some data.
    // Note: According to OpenSSL documentation, unlike `::write()`, a return
    //       value of zero indates failure.
    int res = ::SSL_write(this->m_ssl, data,
                          static_cast<int>(::rocket::min(size, UINT_MAX / 2)));
    if(res > 0)
      return static_cast<IO_Result>(res);  // success

    return do_translate_ssl_error("SSL_write", this->m_ssl, res);
  }

IO_Result
OpenSSL_IO::
shutdown()
  {
    // Perform normal closure.
    int res = ::SSL_shutdown(this->m_ssl);
    if(res == 1)
      return io_result_eof;  // complete

    if(res == 0)
      return io_result_again;  // not complete

    return do_translate_ssl_error("SSL_shutdown", this->m_ssl, res);
  }

}  // namespace poseidon
