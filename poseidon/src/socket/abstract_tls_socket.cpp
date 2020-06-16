// This file is part of Poseidon.
// Copyleft 2020, LH_Mouse. All wrongs reserved.

#include "../precompiled.hpp"
#include "abstract_tls_socket.hpp"
#include "socket_address.hpp"
#include "../utilities.hpp"
#include <netinet/tcp.h>

namespace poseidon {
namespace {

IO_Result
do_translate_ssl_error(const char* func, ::SSL* ssl, int ret)
  {
    int err = ::SSL_get_error(ssl, ret);
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
        // syscall errno
        err = errno;
        if(err == 0)
          return io_result_eof;

        if(err == EINTR)
          return io_result_not_eof;

        POSEIDON_SSL_THROW("Irrecoverable SSL I/O error\n"
                           "[`$1()` returned `$2`: $3]",
                           func, ret, noadl::format_errno(err));

      default:
        POSEIDON_SSL_THROW("Irrecoverable SSL error\n"
                           "[`$1()` returned `$2`]",
                           func, ret);
    }
  }

}  // namespace

Abstract_TLS_Socket::
~Abstract_TLS_Socket()
  {
  }

void
Abstract_TLS_Socket::
do_set_common_options()
  {
    // Disables Nagle algorithm.
    static constexpr int yes[] = { -1 };
    int res = ::setsockopt(this->get_fd(), IPPROTO_TCP, TCP_NODELAY, yes, sizeof(yes));
    ROCKET_ASSERT(res == 0);

    // This can be overwritten if `async_connect()` is called later.
    ::SSL_set_accept_state(this->m_ssl);
  }

void
Abstract_TLS_Socket::
do_stream_preconnect_unlocked()
  {
    ::SSL_set_connect_state(this->m_ssl);
  }

IO_Result
Abstract_TLS_Socket::
do_stream_read_unlocked(void* data, size_t size)
  {
    int nread = ::SSL_read(this->m_ssl, data,
                     static_cast<int>(::rocket::min(size, UINT_MAX / 2)));
    if(nread > 0)
      return static_cast<IO_Result>(nread);

    return do_translate_ssl_error("SSL_read", this->m_ssl, nread);
  }

IO_Result
Abstract_TLS_Socket::
do_stream_write_unlocked(const void* data, size_t size)
  {
    int nwritten = ::SSL_write(this->m_ssl, data,
                        static_cast<int>(::rocket::min(size, UINT_MAX / 2)));
    if(nwritten > 0)
      return static_cast<IO_Result>(nwritten);

    return do_translate_ssl_error("SSL_write", this->m_ssl, nwritten);
  }

IO_Result
Abstract_TLS_Socket::
do_stream_preshutdown_unlocked()
  {
    int ret = ::SSL_shutdown(this->m_ssl);
    if(ret == 0)
      ret = ::SSL_shutdown(this->m_ssl);

    if(ret == 1)
      return io_result_eof;

    return do_translate_ssl_error("SSL_shutdown", this->m_ssl, ret);
  }

void
Abstract_TLS_Socket::
do_on_async_establish()
  {
    POSEIDON_LOG_INFO("Secure TCP connection established: local '$1', remote '$2'",
                      this->get_local_address(), this->get_remote_address());
  }

void
Abstract_TLS_Socket::
do_on_async_shutdown(int err)
  {
    POSEIDON_LOG_INFO("Secure TCP connection closed: local '$1', $2",
                      this->get_local_address(), noadl::format_errno(err));
  }

}  // namespace poseidon
