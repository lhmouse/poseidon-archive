// This file is part of Poseidon.
// Copyleft 2022, LH_Mouse. All wrongs reserved.

#include "../precompiled.ipp"
#include "udp_server_socket.hpp"
#include "../static/async_logger.hpp"
#include "../utils.hpp"
#include <sys/socket.h>

namespace poseidon {

UDP_Server_Socket::
UDP_Server_Socket(const Socket_Address& addr)
  : UDP_Socket(addr.family())
  {
    int ival = 1;
    ::setsockopt(this->fd(), SOL_SOCKET, SO_REUSEADDR, &ival, sizeof(ival));

    if(::bind(this->fd(), addr.addr(), addr.ssize()) != 0)
      POSEIDON_THROW((
          "Failed to bind UDP socket onto `$4`",
          "[`bind()` failed: $3]",
          "[UDP socket `$1` (class `$2`)]"),
          this, typeid(*this), format_errno(), addr);

    POSEIDON_LOG_INFO((
        "UDP server started listening on `$3`",
        "[UDP socket `$1` (class `$2`)]"),
        this, typeid(*this), this->get_local_address());
  }

UDP_Server_Socket::
~UDP_Server_Socket()
  {
  }

void
UDP_Server_Socket::
do_abstract_socket_on_closed(int err)
  {
    POSEIDON_LOG_INFO((
        "UDP server stopped listening on `$3`: $4",
        "[UDP socket `$1` (class `$2`)]"),
        this, typeid(*this), this->get_local_address(), format_errno(err));
  }

}  // namespace poseidon
