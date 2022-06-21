// This file is part of Poseidon.
// Copyleft 2022, LH_Mouse. All wrongs reserved.

#include "../precompiled.ipp"
#include "udp_server_socket.hpp"
#include "../static/async_logger.hpp"
#include "../utils.hpp"

namespace poseidon {

UDP_Server_Socket::
UDP_Server_Socket(const Socket_Address& baddr)
  : UDP_Socket(baddr.family())
  {
    int ival = 1;
    ::setsockopt(this->fd(), SOL_SOCKET, SO_REUSEADDR, &ival, sizeof(ival));

    if(::bind(this->fd(), baddr.addr(), baddr.ssize()) != 0)
        POSEIDON_THROW((
            "Failed to bind socket onto `$4`",
            "[`bind()` failed: $3]"),
            this, typeid(*this), format_errno(), baddr);
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
        "UDP server socket closed: $3",
        "[UDP socket `$1` (class `$2`)]"),
        this, typeid(*this), format_errno(err));
  }

}  // namespace poseidon
