// This file is part of Poseidon.
// Copyleft 2020, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_SOCKET_ABSTRACT_TCP_CLIENT_SOCKET_HPP_
#define POSEIDON_SOCKET_ABSTRACT_TCP_CLIENT_SOCKET_HPP_

#include "abstract_tcp_socket.hpp"

namespace poseidon {

class Abstract_TCP_Client_Socket
  : public ::asteria::Rcfwd<Abstract_TCP_Client_Socket>,
    public Abstract_TCP_Socket
  {
  public:
    explicit
    Abstract_TCP_Client_Socket(const Socket_Address& addr)
      : Abstract_TCP_Socket(addr.create_socket(SOCK_STREAM, IPPROTO_TCP))
      { this->do_async_connect(addr);  }

    Abstract_TCP_Client_Socket(const char* host, uint16_t port)
      : Abstract_TCP_Client_Socket(Socket_Address(host, port))
      { }

    ASTERIA_NONCOPYABLE_DESTRUCTOR(Abstract_TCP_Client_Socket);

  private:
    // This functions is forbidden for derived classes.
    using Abstract_TCP_Socket::do_async_connect;

  public:
    using Abstract_Socket::get_fd;
    using Abstract_Socket::terminate;
    using Abstract_Socket::get_local_address;

    using Abstract_Stream_Socket::get_remote_address;
    using Abstract_Stream_Socket::shut_down;
  };

}  // namespace poseidon

#endif
