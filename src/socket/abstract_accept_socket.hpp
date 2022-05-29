// This file is part of Poseidon.
// Copyleft 2020, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_SOCKET_ABSTRACT_ACCEPT_SOCKET_
#define POSEIDON_SOCKET_ABSTRACT_ACCEPT_SOCKET_

#include "abstract_socket.hpp"

namespace poseidon {

class Abstract_Accept_Socket
  : public ::asteria::Rcfwd<Abstract_Accept_Socket>,
    public Abstract_Socket
  {
  protected:
    // Creates a new non-blocking socket.
    explicit
    Abstract_Accept_Socket(::sa_family_t family);

  protected:
    // Accepts a socket in non-blocking mode.
    // Please mind thread safety, as this function is called by the network thread.
    IO_Result
    do_socket_on_poll_read(simple_mutex::unique_lock& lock) final;

    // Does nothing.
    // This function always returns zero.
    size_t
    do_write_queue_size(simple_mutex::unique_lock& lock) const final;

    // Does nothing.
    // This function always returns `io_result_eof`.
    IO_Result
    do_socket_on_poll_write(simple_mutex::unique_lock& lock) final;

    // Prints a line of text but does nothing otherwise.
    // Please mind thread safety, as this function is called by the network thread.
    void
    do_socket_on_poll_close(int err) final;

    // Binds this socket to the specified address and starts listening.
    // `backlog` is clamped between `1` and `SOMAXCONN`. Out-of-bound values
    // are truncated silently.
    void
    do_socket_listen(const Socket_Address& addr, uint32_t backlog = UINT32_MAX);

    // Consumes an accepted socket descriptor.
    // This function shall allocate and return a new socket object.
    // Please mind thread safety, as this function is called by the network thread.
    virtual
    uptr<Abstract_Socket>
    do_socket_on_accept(unique_FD&& fd, const Socket_Address& addr)
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
    close() noexcept;
  };

}  // namespace poseidon

#endif
