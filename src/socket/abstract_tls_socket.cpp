// This file is part of Poseidon.
// Copyleft 2020, LH_Mouse. All wrongs reserved.

#include "../precompiled.ipp"
#include "abstract_tls_socket.hpp"
#include "../utils.hpp"

namespace poseidon {
namespace {

IO_Result
do_translate_ssl_error(const char* func, const ::SSL* ssl, int ret)
  {
    int err = ::SSL_get_error(ssl, ret);
    switch(err) {
      case SSL_ERROR_NONE:
      case SSL_ERROR_ZERO_RETURN:
        return io_result_end_of_stream;

      case SSL_ERROR_WANT_READ:
      case SSL_ERROR_WANT_WRITE:
      case SSL_ERROR_WANT_CONNECT:
      case SSL_ERROR_WANT_ACCEPT:
        return io_result_would_block;

      case SSL_ERROR_SYSCALL:
        err = errno;
        details_openssl_common::log_openssl_errors();
        return get_io_result_from_errno(func, err);

      default:
        POSEIDON_SSL_THROW("openSSL I/O error\n[`$1()` returned `$2`]",
                           func, ret);
    }
  }

}  // namespace

Abstract_TLS_Socket::
Abstract_TLS_Socket(unique_FD&& fd, const OpenSSL_Context& ctx)
  : Abstract_Stream_Socket(::std::move(fd)),
    OpenSSL_Stream(ctx, *this)
  {
    ::SSL_set_accept_state(this->mut_ssl());
  }

Abstract_TLS_Socket::
Abstract_TLS_Socket(::sa_family_t family, const OpenSSL_Context& ctx)
  : Abstract_Stream_Socket(family),
    OpenSSL_Stream(ctx, *this)
  {
    ::SSL_set_connect_state(this->mut_ssl());
  }

Abstract_TLS_Socket::
~Abstract_TLS_Socket()
  {
  }

IO_Result
Abstract_TLS_Socket::
do_socket_stream_read_unlocked(char*& data, size_t size)
  {
    int nread = ::SSL_read(this->mut_ssl(), data,
                     ::rocket::clamp_cast<int>(size, 0, INT_MAX));
    if(nread < 0)
      return do_translate_ssl_error("SSL_read", this->mut_ssl(), nread);

    if(nread == 0)
      return io_result_end_of_stream;

    data += static_cast<unsigned>(nread);
    return io_result_partial_work;
  }

IO_Result
Abstract_TLS_Socket::
do_socket_stream_write_unlocked(const char*& data, size_t size)
  {
    int nwritten = ::SSL_write(this->mut_ssl(), data,
                        ::rocket::clamp_cast<int>(size, 0, INT_MAX));
    if(nwritten < 0)
      return do_translate_ssl_error("SSL_write", this->mut_ssl(), nwritten);

    data += static_cast<unsigned>(nwritten);
    return io_result_partial_work;
  }

void
Abstract_TLS_Socket::
do_socket_stream_preclose_unclocked() noexcept
  {
    ::SSL_shutdown(this->mut_ssl());
  }

void
Abstract_TLS_Socket::
do_socket_on_establish()
  {
    POSEIDON_LOG_INFO("TLS/TCP connection established: local '$1', remote '$2'",
                      this->get_local_address(), this->get_remote_address());
  }

void
Abstract_TLS_Socket::
do_socket_on_close(int err)
  {
    POSEIDON_LOG_INFO("TLS/TCP connection closed: local '$1', reason: $2",
                      this->get_local_address(), format_errno(err));
  }

}  // namespace poseidon
