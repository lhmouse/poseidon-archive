// This file is part of Poseidon.
// Copyleft 2022, LH_Mouse. All wrongs reserved.

#include "../precompiled.ipp"
#include "udp_client_socket.hpp"
#include "../static/async_logger.hpp"
#include "../utils.hpp"

namespace poseidon {

UDP_Client_Socket::
UDP_Client_Socket()
  : UDP_Socket()
  {
  }

UDP_Client_Socket::
~UDP_Client_Socket()
  {
  }

void
UDP_Client_Socket::
do_abstract_socket_on_closed(int err)
  {
    POSEIDON_LOG_INFO((
        "UDP client socket closed: $3",
        "[UDP socket `$1` (class `$2`)]"),
        this, typeid(*this), format_errno(err));
  }

}  // namespace poseidon
