// This file is part of Poseidon.
// Copyleft 2020, LH_Mouse. All wrongs reserved.

#include "../precompiled.ipp"
#include "abstract_udp_server_socket.hpp"
#include "abstract_udp_socket.hpp"
#include "../utils.hpp"

namespace poseidon {

Abstract_UDP_Server_Socket::
Abstract_UDP_Server_Socket(const Socket_Address& addr)
  : Abstract_UDP_Socket(addr.family())
  {
    this->do_socket_bind(addr);
  }

Abstract_UDP_Server_Socket::
~Abstract_UDP_Server_Socket()
  {
  }

}  // namespace poseidon
