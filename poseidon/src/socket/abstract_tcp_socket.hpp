// This file is part of Poseidon.
// Copyleft 2020, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_SOCKET_ABSTRACT_TCP_SOCKET_HPP_
#define POSEIDON_SOCKET_ABSTRACT_TCP_SOCKET_HPP_

#include "abstract_stream_socket.hpp"

namespace poseidon {

class Abstract_TCP_Socket
  : public ::asteria::Rcfwd<Abstract_TCP_Socket>,
    public Abstract_Stream_Socket
  {
  protected:
    explicit
    Abstract_TCP_Socket(unique_FD&& fd)
      : Abstract_Stream_Socket(::std::move(fd))
      { this->do_set_common_options();  }

  private:
    // Disables Nagle algorithm, etc.
    void
    do_set_common_options();

    // Does nothing as no preparation is needed.
    void
    do_stream_preconnect_unlocked()
      final;

    // Calls `::read()`.
    IO_Result
    do_stream_read_unlocked(char*& data, size_t size)
      final;

    // Calls `::write()`.
    IO_Result
    do_stream_write_unlocked(const char*& data, size_t size)
      final;

    // Does nothing as no preparation is needed.
    // This function always returns `io_result_eof`.
    IO_Result
    do_stream_preclose_unlocked()
      final;

  protected:
    // Notifies a full-duplex channel has been established.
    // The default implementation prints a message but does nothing otherwise.
    // Please mind thread safety, as this function is called by the network thread.
    void
    do_socket_on_establish()
      override;

    // Consumes incoming data.
    // Please mind thread safety, as this function is called by the network thread.
    void
    do_socket_on_receive(linear_buffer&& rqueue)
      override
      = 0;

    // Notifies a full-duplex channel has been closed.
    // The default implementation prints a message but does nothing otherwise.
    // Please mind thread safety, as this function is called by the network thread.
    void
    do_socket_on_close(int err)
      override;

  public:
    ASTERIA_NONCOPYABLE_DESTRUCTOR(Abstract_TCP_Socket);

    using Abstract_Socket::get_fd;
    using Abstract_Socket::terminate;
    using Abstract_Socket::get_local_address;

    using Abstract_Stream_Socket::get_remote_address;
    using Abstract_Stream_Socket::close;
  };

}  // namespace poseidon

#endif
