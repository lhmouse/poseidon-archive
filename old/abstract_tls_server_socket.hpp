// This file is part of Poseidon.
// Copyleft 2020, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_SOCKET_ABSTRACT_TLS_SERVER_SOCKET_
#define POSEIDON_SOCKET_ABSTRACT_TLS_SERVER_SOCKET_

#include "abstract_accept_socket.hpp"
#include "openssl_context.hpp"

namespace poseidon {

class Abstract_TLS_Server_Socket
  : public ::asteria::Rcfwd<Abstract_TLS_Server_Socket>,
    public Abstract_Accept_Socket,
    public OpenSSL_Context
  {
  protected:
    // Create a listening socket that accepts TLS connections over TCP.
    // `cert` and `pkey` shall point to paths to files containing the server
    // certificate and private key, respectively. The overloads that take no
    // `cert` and `pkey` arguments use the default ones in 'main.conf'.
    explicit
    Abstract_TLS_Server_Socket(const Socket_Address& addr);

    explicit
    Abstract_TLS_Server_Socket(const char* bind, uint16_t port)
      : Abstract_TLS_Server_Socket(Socket_Address(bind, port))
      { }

    explicit
    Abstract_TLS_Server_Socket(const Socket_Address& addr,
                               const char* cert, const char* pkey);

    explicit
    Abstract_TLS_Server_Socket(const char* bind, uint16_t port,
                               const char* cert, const char* pkey)
      : Abstract_TLS_Server_Socket(Socket_Address(bind, port), cert, pkey)
      { }

  protected:
    // Implements `Abstract_Accept_Socket`.
    uptr<Abstract_Socket>
    do_socket_on_accept(unique_FD&& fd, const Socket_Address& addr) final;

    void
    do_socket_on_register(rcptr<Abstract_Socket>&& sock) final;

    // Consumes an accepted socket.
    // This function shall allocate and return a new socket object.
    // Please mind thread safety, as this function is called by the network thread.
    virtual
    uptr<Abstract_TLS_Socket>
    do_socket_on_accept_tls(unique_FD&& fd, const Socket_Address& addr)
      = 0;

    // Registers a socket object.
    // This function shall ensure `sock` is not orphaned by storing the pointer
    // somewhere (for example into a user-defined client map).
    // Please mind thread safety, as this function is called by the network thread.
    virtual
    void
    do_socket_on_register_tls(rcptr<Abstract_TLS_Socket>&& sock)
      = 0;

  public:
    ASTERIA_NONCOPYABLE_DESTRUCTOR(Abstract_TLS_Server_Socket);

    using Abstract_Socket::get_fd;
    using Abstract_Socket::kill;
    using Abstract_Socket::get_local_address;
  };

}  // namespace poseidon

#endif
