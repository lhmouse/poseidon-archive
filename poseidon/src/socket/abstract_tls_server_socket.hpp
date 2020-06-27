// This file is part of Poseidon.
// Copyleft 2020, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_SOCKET_ABSTRACT_TLS_SERVER_SOCKET_HPP_
#define POSEIDON_SOCKET_ABSTRACT_TLS_SERVER_SOCKET_HPP_

#include "abstract_listen_socket.hpp"
#include "abstract_tls_socket.hpp"
#include "openssl.hpp"

namespace poseidon {

class Abstract_TLS_Server_Socket
  : public ::asteria::Rcfwd<Abstract_TLS_Server_Socket>,
    public Abstract_Listen_Socket
  {
  private:
    unique_SSL_CTX m_ctx;

  public:
    Abstract_TLS_Server_Socket(const Socket_Address& addr,
                               const char* cert_opt, const char* pkey_opt)
      : Abstract_Listen_Socket(addr.create_socket(SOCK_STREAM, IPPROTO_TCP)),
        m_ctx(noadl::create_server_ssl_ctx(cert_opt, pkey_opt))
      { this->do_listen(addr);  }

    explicit
    Abstract_TLS_Server_Socket(const Socket_Address& addr)
      : Abstract_TLS_Server_Socket(addr,
                                   nullptr, nullptr)  // use default key in 'main.conf'
      { }

    Abstract_TLS_Server_Socket(const char* bind, uint16_t port,
                               const char* cert_opt, const char* pkey_opt)
      : Abstract_TLS_Server_Socket(Socket_Address(bind, port),
                                   cert_opt, pkey_opt)
      { }

    Abstract_TLS_Server_Socket(const char* bind, uint16_t port)
      : Abstract_TLS_Server_Socket(bind, port,
                                   nullptr, nullptr)  // use default key in 'main.conf'
      { }

    ASTERIA_NONCOPYABLE_DESTRUCTOR(Abstract_TLS_Server_Socket);

  private:
    uptr<Abstract_Socket>
    do_on_async_accept(unique_FD&& fd)
    final;

    void
    do_on_async_register(rcptr<Abstract_Socket>&& sock)
    final;

  protected:
    // Consumes an accepted socket.
    // This function shall allocate and return a new socket object.
    // Please mind thread safety, as this function is called by the network thread.
    virtual
    uptr<Abstract_TLS_Socket>
    do_on_async_accept_tls(unique_FD&& fd, ::SSL_CTX* ctx)
      = 0;

    // Registers a socket object.
    // This function shall ensure `sock` is not orphaned by storing the pointer
    // somewhere (for example into a user-defined client map).
    // Please mind thread safety, as this function is called by the network thread.
    virtual
    void
    do_on_async_register_tls(rcptr<Abstract_TLS_Socket>&& sock)
      = 0;

  public:
    using Abstract_Socket::get_fd;
    using Abstract_Socket::abort;
    using Abstract_Socket::get_local_address;
  };

}  // namespace poseidon

#endif
