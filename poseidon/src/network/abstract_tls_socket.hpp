// This file is part of Poseidon.
// Copyleft 2020, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_NETWORK_ABSTRACT_TLS_SOCKET_HPP_
#define POSEIDON_NETWORK_ABSTRACT_TLS_SOCKET_HPP_

#include "abstract_stream_socket.hpp"
#include "openssl.hpp"

namespace poseidon {

class Abstract_TLS_Socket
  : public ::asteria::Rcfwd<Abstract_TLS_Socket>,
    public Abstract_Stream_Socket
  {
  private:
    unique_SSL m_ssl;

  public:
    Abstract_TLS_Socket(unique_FD&& fd, ::SSL_CTX* ctx)
      : Abstract_Stream_Socket(::std::move(fd)),
        m_ssl(noadl::create_ssl(ctx, this->get_fd()))
      { this->do_set_common_options();  }

    ASTERIA_NONCOPYABLE_DESTRUCTOR(Abstract_TLS_Socket);

  private:
    // Disables Nagle algorithm, etc.
    // Calls `::SSL_set_accept_state()`.
    void
    do_set_common_options();

    // Calls `::SSL_set_connect_state()`.
    void
    do_stream_preconnect_nolock()
    final;

    // Calls `::SSL_read()`.
    IO_Result
    do_stream_read_nolock(void* data, size_t size)
    final;

    // Calls `::SSL_write()`.
    IO_Result
    do_stream_write_nolock(const void* data, size_t size)
    final;

    // Calls `::SSL_shutdown()`.
    IO_Result
    do_stream_preshutdown_nolock()
    final;

  protected:
    // Notifies a full-duplex channel has been established.
    // The default implementation prints a message but does nothing otherwise.
    // Please mind thread safety, as this function is called by the network thread.
    void
    do_on_async_establish()
    override;

    // Consumes incoming data.
    // Please mind thread safety, as this function is called by the network thread.
    void
    do_on_async_receive(void* data, size_t size)
    override
      = 0;

    // Notifies a full-duplex channel has been closed.
    // The default implementation prints a message but does nothing otherwise.
    // Please mind thread safety, as this function is called by the network thread.
    void
    do_on_async_shutdown(int err)
    override;

  public:
    using Abstract_Socket::get_fd;
    using Abstract_Socket::abort;
    using Abstract_Socket::get_local_address;

    using Abstract_Stream_Socket::get_remote_address;
    using Abstract_Stream_Socket::async_send;
    using Abstract_Stream_Socket::async_shutdown;
  };

}  // namespace poseidon

#endif
