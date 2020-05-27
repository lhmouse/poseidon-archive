// This file is part of Poseidon.
// Copyleft 2020, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_NETWORK_ABSTRACT_LISTEN_SOCKET_HPP_
#define POSEIDON_NETWORK_ABSTRACT_LISTEN_SOCKET_HPP_

#include "abstract_socket.hpp"

namespace poseidon {

class Abstract_Listen_Socket
  : public Abstract_Socket
  {
  public:
    explicit
    Abstract_Listen_Socket(::rocket::unique_posix_fd&& fd)
      : Abstract_Socket(::std::move(fd))
      { this->do_set_common_options();  }

    ASTERIA_NONCOPYABLE_DESTRUCTOR(Abstract_Listen_Socket);

  private:
    // Enables `SO_REUSEADDR`, etc.
    void
    do_set_common_options();

    // Accepts a socket in non-blocking mode.
    // `lock` and `hint` are ignored.
    // Please mind thread safety, as this function is called by the network thread.
    IO_Result
    do_on_async_poll_read(Si_Mutex::unique_lock& lock, void* hint, size_t size)
    final;

    // Does nothing.
    // This function always returns zero.
    // `lock` is ignored.
    size_t
    do_write_queue_size(Si_Mutex::unique_lock& lock)
    const final;

    // Does nothing.
    // This function always returns `io_result_eof`.
    // `lock` is ignored.
    IO_Result
    do_on_async_poll_write(Si_Mutex::unique_lock& lock, void* hint, size_t size)
    final;

  protected:
    // Consumes an accepted socket.
    // This function shall allocate and return a new socket object.
    // Please mind thread safety, as this function is called by the network thread.
    virtual
    uptr<Abstract_Socket>
    do_on_async_accept(unique_posix_fd&& fd)
      = 0;

    // Prints a line of text but does nothing otherwise.
    // Please mind thread safety, as this function is called by the network thread.
    void
    do_on_async_poll_shutdown(int err)
    override;

  public:
    using Abstract_Socket::get_fd;
    using Abstract_Socket::abort;
    using Abstract_Socket::get_local_address;

    // Binds this socket to the specified address and starts listening.
    // `backlog` is clamped between `1` and `SOMAXCONN`. Out-of-bound values
    // are truncated silently.
    void
    listen(const Socket_Address& addr, uint32_t backlog = UINT32_MAX);
  };

}  // namespace poseidon

#endif
