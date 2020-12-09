// This file is part of Poseidon.
// Copyleft 2020, LH_Mouse. All wrongs reserved.

#include "../precompiled.hpp"
#include "abstract_tls_client_socket.hpp"
#include "../utils.hpp"

namespace poseidon {

Abstract_TLS_Client_Socket::
Abstract_TLS_Client_Socket(const Socket_Address& addr)
  : Abstract_TLS_Socket(addr.family(), get_client_ssl_ctx())
  {
    this->do_socket_connect(addr);
  }

Abstract_TLS_Client_Socket::
~Abstract_TLS_Client_Socket()
  {
  }

}  // namespace poseidon
