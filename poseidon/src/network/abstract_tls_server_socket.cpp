// This file is part of Poseidon.
// Copyleft 2020, LH_Mouse. All wrongs reserved.

#include "../precompiled.hpp"
#include "abstract_tls_server_socket.hpp"
#include "../utilities.hpp"

namespace poseidon {

Abstract_TLS_Server_Socket::
~Abstract_TLS_Server_Socket()
  {
  }

uptr<Abstract_Socket>
Abstract_TLS_Server_Socket::
do_on_async_accept(unique_FD&& fd)
  {
    return this->do_on_async_accept_tls(::std::move(fd), this->m_ctx);
  }

}  // namespace poseidon
