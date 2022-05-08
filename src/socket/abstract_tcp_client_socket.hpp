// This file is part of Poseidon.
// Copyleft 2020, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_SOCKET_ABSTRACT_TCP_CLIENT_SOCKET_
#define POSEIDON_SOCKET_ABSTRACT_TCP_CLIENT_SOCKET_

#include "abstract_tcp_socket.hpp"

namespace poseidon {

class Abstract_TCP_Client_Socket
  : public ::asteria::Rcfwd<Abstract_TCP_Client_Socket>,
    public Abstract_TCP_Socket
  {
  protected:
    // Creates a TCP socket that will connect to `addr`.
    explicit
    Abstract_TCP_Client_Socket(const Socket_Address& addr);

    explicit
    Abstract_TCP_Client_Socket(const char* host, uint16_t port)
      : Abstract_TCP_Client_Socket(Socket_Address(host, port))
      { }

  private:
    // This functions is forbidden for derived classes.
    using Abstract_TCP_Socket::do_socket_connect;

  public:
    ASTERIA_NONCOPYABLE_DESTRUCTOR(Abstract_TCP_Client_Socket);

    using Abstract_Socket::get_fd;
    using Abstract_Socket::kill;
    using Abstract_Socket::get_local_address;

    using Abstract_Stream_Socket::get_remote_address;
    using Abstract_Stream_Socket::close;
  };

}  // namespace poseidon

#endif
