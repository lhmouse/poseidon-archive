// This file is part of Poseidon.
// Copyleft 2022, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_SOCKET_SSL_SOCKET_
#define POSEIDON_SOCKET_SSL_SOCKET_

#include "../fwd.hpp"
#include "abstract_socket.hpp"
#include "ssl_ptr.hpp"
#include "../core/charbuf_256.hpp"

namespace poseidon {

class SSL_Socket
  : public Abstract_Socket
  {
  private:
    SSL_ptr m_ssl;

    mutable atomic_acq_rel<bool> m_peername_ready;
    mutable plain_mutex m_peername_mutex;
    mutable cow_string m_peername;

  protected:
    // Server-side constructor:
    // Takes ownership of an accepted socket.
    explicit
    SSL_Socket(unique_posix_fd&& fd, const SSL_CTX_ptr& ssl_ctx);

    // Client-side constructor:
    // Creates a new non-blocking socket to the target host.
    explicit
    SSL_Socket(const Socket_Address& addr, const SSL_CTX_ptr& ssl_ctx);

  protected:
    // These callbacks implement `Abstract_Socket`.
    virtual
    void
    do_abstract_socket_on_closed(int err) override;

    virtual
    void
    do_abstract_socket_on_readable() override;

    virtual
    void
    do_abstract_socket_on_writable() override;

    virtual
    void
    do_abstract_socket_on_exception(exception& stdex) override;

    // This callback is invoked by the network thread when an outgoing (from
    // client) full-duplex connection has been established. It is not called for
    // incoming connections.
    // The default implemention merely prints a message.
    virtual
    void
    do_on_ssl_established();

    // This callback is invoked by the network thread when some bytes have been
    // received, and is intended to be overriden by derived classes.
    // The argument contains all data that have been accumulated so far. Callees
    // should remove processed bytes.
    virtual
    void
    do_on_ssl_stream(linear_buffer& data)
      = 0;

    // Prepares a list of protocols that will be sent to the server for
    // Application-Layer Protocol Negotiation (ALPN). This is meaningful only on
    // a client-side socket. If ALPN is desired, this function shall be called
    // before adding this socket into a network driver.
    // The argument is the list of names of protocols that will be sent. Empty
    // protocol names are ignored. If the list is empty, ALPN is not requested.
    void
    do_ssl_alpn_request(const charbuf_256* protos_opt, size_t protos_size);

    void
    do_ssl_alpn_request(const cow_vector<charbuf_256>& protos);

    void
    do_ssl_alpn_request(initializer_list<charbuf_256> protos);

    void
    do_ssl_alpn_request(const charbuf_256& proto);

  public:
    ASTERIA_NONCOPYABLE_VIRTUAL_DESTRUCTOR(SSL_Socket);

    // Gets the SSL structure.
    ::SSL*
    ssl() const noexcept
      { return this->m_ssl.get();  }

    // Gets the remote or connected address of this socket as a human-readable
    // string. In case of errors, a string with information about the error is
    // returned instead.
    const cow_string&
    get_remote_address() const;

    // Enqueues some bytes for sending.
    // The return value merely indicates whether the attempt has succeeded. The
    // bytes may or may never arrive at the destination host.
    // This function is thread-safe.
    bool
    ssl_send(const char* data, size_t size);

    bool
    ssl_send(const linear_buffer& data);

    bool
    ssl_send(const cow_string& data);

    bool
    ssl_send(const string& data);

    // Shuts the socket down gracefully.
    // This function is thread-safe.
    bool
    ssl_shut_down() noexcept;
  };

}  // namespace poseidon

#endif
