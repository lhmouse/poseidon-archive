// This file is part of Poseidon.
// Copyleft 2020, LH_Mouse. All wrongs reserved.

#include "../precompiled.hpp"
#include "abstract_tls_socket.hpp"
#include "../utils.hpp"

namespace poseidon {

Abstract_TLS_Socket::
Abstract_TLS_Socket(unique_FD&& fd, ::SSL_CTX* ctx)
  : Abstract_Stream_Socket(::std::move(fd)),
    m_ssl(create_ssl(ctx, this->get_fd()))
  {
  }

Abstract_TLS_Socket::
Abstract_TLS_Socket(::sa_family_t family, ::SSL_CTX* ctx)
  : Abstract_Stream_Socket(family),
    m_ssl(create_ssl(ctx, this->get_fd()))
  {
  }

Abstract_TLS_Socket::
~Abstract_TLS_Socket()
  {
  }

void
Abstract_TLS_Socket::
do_stream_preconnect_unlocked()
  {
    ::SSL_set_connect_state(this->m_ssl);
  }

IO_Result
Abstract_TLS_Socket::
do_stream_read_unlocked(char*& data, size_t size)
  {
    int nread = ::SSL_read(this->m_ssl, data,
                     static_cast<int>(::rocket::min(size, UINT_MAX / 2)));
    if(nread < 0)
      return get_io_result_from_ssl_error("SSL_read", this->m_ssl, nread);

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
                        static_cast<int>(::rocket::min(size, UINT_MAX / 2)));
    if(nwritten < 0)
      return get_io_result_from_ssl_error("SSL_write", this->m_ssl, nwritten);

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
      return get_io_result_from_ssl_error("SSL_shutdown", this->m_ssl, ret);

    return io_result_end_of_stream;
  }

void
Abstract_TLS_Socket::
do_socket_on_establish()
  {
    POSEIDON_LOG_INFO("Secure TCP connection established: local '$1', remote '$2'",
                      this->get_local_address(), this->get_remote_address());
  }

void
Abstract_TLS_Socket::
do_socket_on_close(int err)
  {
    POSEIDON_LOG_INFO("Secure TCP connection closed: local '$1', $2",
                      this->get_local_address(), format_errno(err));
  }

}  // namespace poseidon
