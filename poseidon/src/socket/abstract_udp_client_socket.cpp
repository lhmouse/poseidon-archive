// This file is part of Poseidon.
// Copyleft 2020, LH_Mouse. All wrongs reserved.

#include "../precompiled.hpp"
#include "abstract_udp_client_socket.hpp"
#include "../utils.hpp"

namespace poseidon {

Abstract_UDP_Client_Socket::
Abstract_UDP_Client_Socket(const Socket_Address& addr)
  : Abstract_UDP_Socket(addr.family())
  {
  }

Abstract_UDP_Client_Socket::
~Abstract_UDP_Client_Socket()
  {
  }

}  // namespace poseidon
