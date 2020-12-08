// This file is part of Poseidon.
// Copyleft 2020, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_SOCKET_ABSTRACT_TCP_SERVER_SOCKET_HPP_
#define POSEIDON_SOCKET_ABSTRACT_TCP_SERVER_SOCKET_HPP_

#include "abstract_accept_socket.hpp"

namespace poseidon {

class Abstract_TCP_Server_Socket
  : public ::asteria::Rcfwd<Abstract_TCP_Server_Socket>,
    public Abstract_Accept_Socket
  {
  protected:
    explicit
    Abstract_TCP_Server_Socket(const Socket_Address& addr)
      : Abstract_Accept_Socket(addr.create_socket(SOCK_STREAM, IPPROTO_TCP))
      { this->do_listen(addr);  }

    Abstract_TCP_Server_Socket(const char* bind, uint16_t port)
      : Abstract_TCP_Server_Socket(Socket_Address(bind, port))
      { }

  private:
    uptr<Abstract_Socket>
    do_socket_on_accept(unique_FD&& fd)
      final;

    void
    do_socket_on_register(rcptr<Abstract_Socket>&& sock)
      final;

  protected:
    // Consumes an accepted socket.
    // This function shall allocate and return a new socket object.
    // Please mind thread safety, as this function is called by the network thread.
    virtual
    uptr<Abstract_TCP_Socket>
    do_socket_on_accept_tcp(unique_FD&& fd)
      = 0;

    // Registers a socket object.
    // This function shall ensure `sock` is not orphaned by storing the pointer
    // somewhere (for example into a user-defined client map).
    // Please mind thread safety, as this function is called by the network thread.
    virtual
    void
    do_socket_on_register_tcp(rcptr<Abstract_TCP_Socket>&& sock)
      = 0;

  public:
    ASTERIA_NONCOPYABLE_DESTRUCTOR(Abstract_TCP_Server_Socket);

    using Abstract_Socket::get_fd;
    using Abstract_Socket::kill;
    using Abstract_Socket::get_local_address;
  };

}  // namespace poseidon

#endif
