// This file is part of Poseidon.
// Copyleft 2020, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_NETWORK_ABSTRACT_TLS_TCP_SOCKET_HPP_
#define POSEIDON_NETWORK_ABSTRACT_TLS_TCP_SOCKET_HPP_

#include "abstract_stream_socket.hpp"
#include <openssl/ssl.h>

namespace poseidon {

class Abstract_TLS_TCP_Socket
  : public Abstract_Stream_Socket
  {
  public:
    using base_type = Abstract_Stream_Socket;

  private:
    uptr<::SSL, void (&)(::SSL*)> m_ssl;

  public:
    Abstract_TLS_TCP_Socket(unique_posix_fd&& fd, uptr<::SSL, void (&)(::SSL*)>&& ssl)
      : base_type(::std::move(fd)),
        m_ssl(::std::move(ssl))
      { this->do_set_common_options();  }

    ASTERIA_NONCOPYABLE_DESTRUCTOR(Abstract_TLS_TCP_Socket);

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
    // These are functions inherited from `Abstract_Socket`.
    using base_type::get_fd;
    using base_type::abort;
    using base_type::get_local_address;
    using base_type::get_remote_address;

    // These are functions inherited from `Abstract_Stream_Socket`.
    using base_type::async_send;
    using base_type::async_shutdown;
  };

}  // namespace poseidon

#endif
