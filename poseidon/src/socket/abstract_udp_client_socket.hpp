// This file is part of Poseidon.
// Copyleft 2020, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_SOCKET_ABSTRACT_UDP_CLIENT_SOCKET_HPP_
#define POSEIDON_SOCKET_ABSTRACT_UDP_CLIENT_SOCKET_HPP_

#include "abstract_udp_socket.hpp"
#include "abstract_udp_socket.hpp"

namespace poseidon {

class Abstract_UDP_Client_Socket
  : public ::asteria::Rcfwd<Abstract_UDP_Client_Socket>,
    public Abstract_UDP_Socket
  {
  public:
    explicit
    Abstract_UDP_Client_Socket(const Socket_Address& addr)
      : Abstract_UDP_Socket(addr.create_socket(SOCK_DGRAM, IPPROTO_UDP))
      { }

    Abstract_UDP_Client_Socket(const char* host, uint16_t port)
      : Abstract_UDP_Client_Socket(Socket_Address(host, port))
      { }

    ASTERIA_NONCOPYABLE_DESTRUCTOR(Abstract_UDP_Client_Socket);

  protected:
    // Consumes an incoming packet.
    // Please mind thread safety, as this function is called by the network thread.
    virtual
    void
    do_on_socket_receive(Socket_Address&& addr, char* data, size_t size)
      = 0;

  public:
    using Abstract_Socket::get_fd;
    using Abstract_Socket::terminate;
    using Abstract_Socket::get_local_address;
  };

}  // namespace poseidon

#endif
