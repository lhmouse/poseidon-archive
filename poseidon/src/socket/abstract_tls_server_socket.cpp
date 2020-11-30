// This file is part of Poseidon.
// Copyleft 2020, LH_Mouse. All wrongs reserved.

#include "../precompiled.hpp"
#include "abstract_tls_server_socket.hpp"
#include "abstract_tls_socket.hpp"
#include "../util.hpp"

namespace poseidon {

Abstract_TLS_Server_Socket::
~Abstract_TLS_Server_Socket()
  {
  }

uptr<Abstract_Socket>
Abstract_TLS_Server_Socket::
do_on_socket_accept(unique_FD&& fd)
  {
    return this->do_on_socket_accept_tls(::std::move(fd), this->m_ctx);
  }

void
Abstract_TLS_Server_Socket::
do_on_socket_register(rcptr<Abstract_Socket>&& sock)
  {
    return this->do_on_socket_register_tls(
        ::rocket::static_pointer_cast<Abstract_TLS_Socket>(::std::move(sock)));
  }

}  // namespace poseidon
