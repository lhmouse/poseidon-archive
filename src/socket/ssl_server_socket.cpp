// This file is part of Poseidon.
// Copyleft 2022, LH_Mouse. All wrongs reserved.

#include "../precompiled.ipp"
#include "ssl_server_socket.hpp"
#include "../static/async_logger.hpp"
#include "../utils.hpp"

namespace poseidon {

SSL_Server_Socket::
SSL_Server_Socket(unique_posix_fd&& fd, const SSL_CTX_ptr& ssl_ctx)
  : SSL_Socket(::std::move(fd), ssl_ctx)
  {
    POSEIDON_LOG_INFO((
        "New client connected from `$3`",
        "[SSL socket `$1` (class `$2`)]"),
        this, typeid(*this), this->get_remote_address());
  }

SSL_Server_Socket::
~SSL_Server_Socket()
  {
  }

void
SSL_Server_Socket::
do_abstract_socket_on_closed(int err)
  {
    POSEIDON_LOG_INFO((
        "Client disconnected from `$3`: $4",
        "[SSL socket `$1` (class `$2`)]"),
        this, typeid(*this), this->get_remote_address(), format_errno(err));
  }

}  // namespace poseidon
