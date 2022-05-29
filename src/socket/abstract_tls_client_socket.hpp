// This file is part of Poseidon.
// Copyleft 2020, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_SOCKET_ABSTRACT_TLS_CLIENT_SOCKET_
#define POSEIDON_SOCKET_ABSTRACT_TLS_CLIENT_SOCKET_

#include "abstract_tls_socket.hpp"

namespace poseidon {

class Abstract_TLS_Client_Socket
  : public ::asteria::Rcfwd<Abstract_TLS_Client_Socket>,
    public Abstract_TLS_Socket
  {
  protected:
    // Creates a TCP socket that will connect to `addr`.
    // If no SSL context is specified, the standard peer verfication
    // context is used for security reasons.
    explicit
    Abstract_TLS_Client_Socket(const Socket_Address& addr);

    explicit
    Abstract_TLS_Client_Socket(const char* host, uint16_t port)
      : Abstract_TLS_Client_Socket(Socket_Address(host, port))
      { }

    explicit
    Abstract_TLS_Client_Socket(const Socket_Address& addr,
                               const OpenSSL_Context& ctx);

    explicit
    Abstract_TLS_Client_Socket(const char* host, uint16_t port,
                               const OpenSSL_Context& ctx)
      : Abstract_TLS_Client_Socket(Socket_Address(host, port), ctx)
      { }

  public:
    ASTERIA_NONCOPYABLE_DESTRUCTOR(Abstract_TLS_Client_Socket);

    using Abstract_Socket::get_fd;
    using Abstract_Socket::kill;
    using Abstract_Socket::get_local_address;

    using Abstract_Stream_Socket::get_remote_address;
    using Abstract_Stream_Socket::close;
  };

}  // namespace poseidon

#endif
