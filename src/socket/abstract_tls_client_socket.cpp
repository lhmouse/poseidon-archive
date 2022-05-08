// This file is part of Poseidon.
// Copyleft 2020, LH_Mouse. All wrongs reserved.

#include "../precompiled.ipp"
#include "abstract_tls_client_socket.hpp"
#include "openssl_context.hpp"
#include "../utils.hpp"

namespace poseidon {

Abstract_TLS_Client_Socket::
Abstract_TLS_Client_Socket(const Socket_Address& addr)
  : Abstract_TLS_Socket(addr.family(), OpenSSL_Context::static_verify_peer())
  {
    this->do_socket_connect(addr);
  }

Abstract_TLS_Client_Socket::
Abstract_TLS_Client_Socket(const Socket_Address& addr, const OpenSSL_Context& ctx)
  : Abstract_TLS_Socket(addr.family(), ctx)
  {
    this->do_socket_connect(addr);
  }

Abstract_TLS_Client_Socket::
~Abstract_TLS_Client_Socket()
  {
  }

}  // namespace poseidon
