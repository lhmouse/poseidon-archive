// This file is part of Poseidon.
// Copyleft 2020, LH_Mouse. All wrongs reserved.

#include "../precompiled.hpp"
#include "abstract_tls_server_socket.hpp"
#include "abstract_tls_socket.hpp"
#include "../utils.hpp"

namespace poseidon {

Abstract_TLS_Server_Socket::
Abstract_TLS_Server_Socket(const Socket_Address& addr,
                           const char* cert_opt, const char* pkey_opt)
  : Abstract_Accept_Socket(addr.family()),
    m_ctx(noadl::create_server_ssl_ctx(cert_opt, pkey_opt))
  {
    this->do_socket_listen(addr);
  }

Abstract_TLS_Server_Socket::
~Abstract_TLS_Server_Socket()
  {
  }

uptr<Abstract_Socket>
Abstract_TLS_Server_Socket::
do_socket_on_accept(unique_FD&& fd)
  {
    return this->do_socket_on_accept_tls(::std::move(fd), this->m_ctx);
  }

void
Abstract_TLS_Server_Socket::
do_socket_on_register(rcptr<Abstract_Socket>&& sock)
  {
    return this->do_socket_on_register_tls(
        ::rocket::static_pointer_cast<Abstract_TLS_Socket>(::std::move(sock)));
  }

}  // namespace poseidon
