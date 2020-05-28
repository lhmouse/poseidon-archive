// This file is part of Poseidon.
// Copyleft 2020, LH_Mouse. All wrongs reserved.

#include "../precompiled.hpp"
#include "abstract_tcp_server_socket.hpp"
#include "../utilities.hpp"

namespace poseidon {

Abstract_TCP_Server_Socket::
~Abstract_TCP_Server_Socket()
  {
  }

uptr<Abstract_Socket>
Abstract_TCP_Server_Socket::
do_on_async_accept(unique_FD&& fd)
  {
    return this->do_on_async_accept_tcp(::std::move(fd));
  }

void
Abstract_TCP_Server_Socket::
do_on_async_register(rcptr<Abstract_Socket>&& sock)
  {
    return this->do_on_async_register_tcp(
        ::rocket::static_pointer_cast<Abstract_TCP_Socket>(::std::move(sock)));
  }

}  // namespace poseidon
