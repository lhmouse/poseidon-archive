// This file is part of Poseidon.
// Copyleft 2022, LH_Mouse. All wrongs reserved.

#include "../precompiled.ipp"
#include "tcp_client_socket.hpp"
#include "../static/async_logger.hpp"
#include "../utils.hpp"

namespace poseidon {

TCP_Client_Socket::
TCP_Client_Socket(const Socket_Address& addr)
  : TCP_Socket(addr.family())
  {
    if((::connect(this->fd(), addr.addr(), addr.ssize()) != 0) && (errno != EINPROGRESS))
      POSEIDON_THROW((
          "Failed to initiate TCP connection to `$4`",
          "[`connect()` failed: $3]",
          "[TCP socket `$1` (class `$2`)]"),
          this, typeid(*this), format_errno(), addr);

    POSEIDON_LOG_INFO((
        "Establishing new connection to `$3`",
        "[TCP socket `$1` (class `$2`)]"),
        this, typeid(*this), addr);
  }

TCP_Client_Socket::
~TCP_Client_Socket()
  {
  }

void
TCP_Client_Socket::
do_abstract_socket_on_closed(int err)
  {
    POSEIDON_LOG_INFO((
        "Connection to `$3` closed: $4",
        "[TCP socket `$1` (class `$2`)]"),
        this, typeid(*this), this->get_remote_address(), format_errno(err));
  }

}  // namespace poseidon
