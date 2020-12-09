// This file is part of Poseidon.
// Copyleft 2020, LH_Mouse. All wrongs reserved.

#include "../precompiled.hpp"
#include "abstract_tls_socket.hpp"
#include "../utils.hpp"
#include <openssl/err.h>

namespace poseidon {
namespace {

void
do_dump_ssl_errors()
  noexcept
  {
    char sbuf[512];
    long index = -1;

    while(unsigned long err = ::ERR_get_error())
      ::ERR_error_string_n(err, sbuf, sizeof(sbuf)),
        POSEIDON_LOG_WARN("OpenSSL error: [$1] $2", ++index, sbuf);
  }

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
        do_dump_ssl_errors();
        return get_io_result_from_errno(func, err);

      default:
        do_dump_ssl_errors();
        POSEIDON_THROW("OpenSSL I/O error\n[`$1()` returned `$2`]",
                       func, ret);
    }
  }

}  // namespace

Abstract_TLS_Socket::
Abstract_TLS_Socket(unique_FD&& fd, ::SSL_CTX* ctx)
  : Abstract_Stream_Socket(::std::move(fd)),
    m_ssl(create_ssl(ctx, this->get_fd()))
  {
    ::SSL_set_accept_state(this->m_ssl);
  }

Abstract_TLS_Socket::
Abstract_TLS_Socket(::sa_family_t family, ::SSL_CTX* ctx)
  : Abstract_Stream_Socket(family),
    m_ssl(create_ssl(ctx, this->get_fd()))
  {
    ::SSL_set_connect_state(this->m_ssl);
  }

Abstract_TLS_Socket::
~Abstract_TLS_Socket()
  {
  }

IO_Result
Abstract_TLS_Socket::
do_stream_read_unlocked(char*& data, size_t size)
  {
    int nread = ::SSL_read(this->m_ssl, data,
                    static_cast<int>(::std::min<size_t>(size, INT_MAX)));
    if(nread < 0)
      return do_translate_ssl_error("SSL_read", this->m_ssl, nread);

    if(nread == 0)
      return io_result_end_of_stream;

    data += static_cast<unsigned>(nread);
    return io_result_partial_work;
  }

IO_Result
Abstract_TLS_Socket::
do_stream_write_unlocked(const char*& data, size_t size)
  {
    int nwritten = ::SSL_write(this->m_ssl, data,
                       static_cast<int>(::std::min<size_t>(size, INT_MAX)));
    if(nwritten < 0)
      return do_translate_ssl_error("SSL_write", this->m_ssl, nwritten);

    data += static_cast<unsigned>(nwritten);
    return io_result_partial_work;
  }

IO_Result
Abstract_TLS_Socket::
do_stream_preclose_unlocked()
  {
    int ret = ::SSL_shutdown(this->m_ssl);
    if(ret == 0)
      ret = ::SSL_shutdown(this->m_ssl);

    if(ret < 0)
      return do_translate_ssl_error("SSL_shutdown", this->m_ssl, ret);

    return io_result_end_of_stream;
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
