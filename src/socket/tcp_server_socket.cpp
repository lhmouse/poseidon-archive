// This file is part of Poseidon.
// Copyleft 2022, LH_Mouse. All wrongs reserved.

#include "../precompiled.ipp"
#include "tcp_server_socket.hpp"
#include "../static/async_logger.hpp"
#include "../utils.hpp"

namespace poseidon {

TCP_Server_Socket::
TCP_Server_Socket(unique_posix_fd&& fd)
  : TCP_Socket(::std::move(fd))
  {
    POSEIDON_LOG_INFO((
        "New client connected from `$3`",
        "[TCP socket `$1` (class `$2`)]"),
        this, typeid(*this), this->get_remote_address());
  }

TCP_Server_Socket::
~TCP_Server_Socket()
  {
  }

void
TCP_Server_Socket::
do_abstract_socket_on_closed(int err)
  {
    POSEIDON_LOG_INFO((
        "Client disconnected from `$3`: $4",
        "[TCP socket `$1` (class `$2`)]"),
        this, typeid(*this), this->get_remote_address(), format_errno(err));
  }

}  // namespace poseidon
