// This file is part of Poseidon.
// Copyleft 2020, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_SOCKET_ABSTRACT_UDP_SERVER_SOCKET_
#define POSEIDON_SOCKET_ABSTRACT_UDP_SERVER_SOCKET_

#include "abstract_udp_socket.hpp"

namespace poseidon {

class Abstract_UDP_Server_Socket
  : public ::asteria::Rcfwd<Abstract_UDP_Server_Socket>,
    public Abstract_UDP_Socket
  {
  protected:
    // Creates a listening socket that accepts UDP datagrams.
    explicit
    Abstract_UDP_Server_Socket(const Socket_Address& addr);

    explicit
    Abstract_UDP_Server_Socket(const char* bind, uint16_t port)
      : Abstract_UDP_Server_Socket(Socket_Address(bind, port))
      { }

  protected:
    // Consumes an incoming packet.
    // Please mind thread safety, as this function is called by the network thread.
    virtual
    void
    do_socket_on_receive(const Socket_Address& addr, char* data, size_t size)
      = 0;

  public:
    ASTERIA_NONCOPYABLE_DESTRUCTOR(Abstract_UDP_Server_Socket);

    using Abstract_Socket::get_fd;
    using Abstract_Socket::kill;
    using Abstract_Socket::get_local_address;
  };

}  // namespace poseidon

#endif
