// This file is part of Poseidon.
// Copyleft 2020, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_SOCKET_ABSTRACT_ACCEPT_SOCKET_HPP_
#define POSEIDON_SOCKET_ABSTRACT_ACCEPT_SOCKET_HPP_

#include "abstract_socket.hpp"

namespace poseidon {

class Abstract_Accept_Socket
  : public ::asteria::Rcfwd<Abstract_Accept_Socket>,
    public Abstract_Socket
  {
  private:
    // These are I/O components.
    mutable simple_mutex m_io_mutex;
    Connection_State m_cstate = connection_state_empty;

  protected:
    // Creates a new non-blocking socket.
    explicit
    Abstract_Accept_Socket(::sa_family_t family);

  private:
    // Accepts a socket in non-blocking mode.
    // `hint` and `size` are ignored.
    // Please mind thread safety, as this function is called by the network thread.
    IO_Result
    do_socket_on_poll_read(simple_mutex::unique_lock& lock, char* hint, size_t size)
      final;

    // Does nothing.
    // This function always returns zero.
    size_t
    do_write_queue_size(simple_mutex::unique_lock& lock)
      const final;

    // Does nothing.
    // This function always returns `io_result_eof`.
    IO_Result
    do_socket_on_poll_write(simple_mutex::unique_lock& lock, char* hint, size_t size)
      final;

    // Prints a line of text but does nothing otherwise.
    // Please mind thread safety, as this function is called by the network thread.
    void
    do_socket_on_poll_close(int err)
      final;

  protected:
    // Binds this socket to the specified address and starts listening.
    // `backlog` is clamped between `1` and `SOMAXCONN`. Out-of-bound values
    // are truncated silently.
    void
    do_socket_listen(const Socket_Address& addr, int backlog = INT_MAX);

    // Consumes an accepted socket descriptor.
    // This function shall allocate and return a new socket object.
    // Please mind thread safety, as this function is called by the network thread.
    virtual
    uptr<Abstract_Socket>
    do_socket_on_accept(unique_FD&& fd)
      = 0;

    // Registers a socket object.
    // This function shall ensure `sock` is not orphaned by storing the pointer
    // somewhere (for example into a user-defined client map).
    // Please mind thread safety, as this function is called by the network thread.
    virtual
    void
    do_socket_on_register(rcptr<Abstract_Socket>&& sock)
      = 0;

    // Notifies that this socket has been fully closed.
    // The default implementation prints a message but does nothing otherwise.
    // Please mind thread safety, as this function is called by the network thread.
    virtual
    void
    do_socket_on_close(int err);

  public:
    ASTERIA_NONCOPYABLE_DESTRUCTOR(Abstract_Accept_Socket);

    // Marks this socket as closed immediately.
    bool
    close()
      noexcept;
  };

}  // namespace poseidon

#endif
