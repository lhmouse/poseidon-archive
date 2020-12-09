// This file is part of Poseidon.
// Copyleft 2020, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_SOCKET_ABSTRACT_TLS_SERVER_SOCKET_HPP_
#define POSEIDON_SOCKET_ABSTRACT_TLS_SERVER_SOCKET_HPP_

#include "abstract_accept_socket.hpp"
#include "openssl.hpp"

namespace poseidon {

class Abstract_TLS_Server_Socket
  : public ::asteria::Rcfwd<Abstract_TLS_Server_Socket>,
    public Abstract_Accept_Socket
  {
  private:
    unique_SSL_CTX m_ctx;

  protected:
    // Creates a listening socket that accepts TLS connections over TCP.
    // If both `cert_opt` and `pkey_opt` are null, the default certificate and
    // private key in 'main.conf' are used.
    explicit
    Abstract_TLS_Server_Socket(const Socket_Address& addr,
                               const char* cert_opt, const char* pkey_opt);

    explicit
    Abstract_TLS_Server_Socket(const char* bind, uint16_t port,
                               const char* cert_opt, const char* pkey_opt)
      : Abstract_TLS_Server_Socket(Socket_Address(bind, port), cert_opt, pkey_opt)
      { }

    explicit
    Abstract_TLS_Server_Socket(const Socket_Address& addr)
      : Abstract_TLS_Server_Socket(addr, nullptr, nullptr)
      { }

    explicit
    Abstract_TLS_Server_Socket(const char* bind, uint16_t port)
      : Abstract_TLS_Server_Socket(Socket_Address(bind, port))
      { }

  private:
    uptr<Abstract_Socket>
    do_socket_on_accept(unique_FD&& fd)
      final;

    void
    do_socket_on_register(rcptr<Abstract_Socket>&& sock)
      final;

  protected:
    // Consumes an accepted socket.
    // This function shall allocate and return a new socket object.
    // Please mind thread safety, as this function is called by the network thread.
    virtual
    uptr<Abstract_TLS_Socket>
    do_socket_on_accept_tls(unique_FD&& fd, ::SSL_CTX* ctx)
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
