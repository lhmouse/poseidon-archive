// This file is part of Poseidon.
// Copyleft 2020, LH_Mouse. All wrongs reserved.

#include "../precompiled.ipp"
#include "abstract_tcp_client_socket.hpp"
#include "../utils.hpp"

namespace poseidon {

Abstract_TCP_Client_Socket::
Abstract_TCP_Client_Socket(const Socket_Address& addr)
  : Abstract_TCP_Socket(addr.family())
  {
    this->do_socket_connect(addr);
  }

Abstract_TCP_Client_Socket::
~Abstract_TCP_Client_Socket()
  {
  }

}  // namespace poseidon
