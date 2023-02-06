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
    friend class Network_Driver;

    SSL_ptr m_ssl;
    cow_string m_alpn_proto;

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
    SSL_Socket(const Socket_Address& saddr, const SSL_CTX_ptr& ssl_ctx);

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
    do_abstract_socket_on_oob_readable() override;

    virtual
    void
    do_abstract_socket_on_writable() override;

    // This callback is invoked by the network thread when an outgoing (from
    // client) full-duplex connection has been established. It is not called for
    // incoming connections.
    // The default implemention merely prints a message.
    virtual
    void
    do_on_ssl_connected();

    // This callback is invoked by the network thread when some bytes have been
    // received, and is intended to be overriden by derived classes.
    // The argument contains all data that have been accumulated so far. Callees
    // should remove processed bytes.
    virtual
    void
    do_on_ssl_stream(linear_buffer& data) = 0;

    // This callback is invoked by the network thread when an out-of-band byte
    // has been received, and is intended to be overriden by derived classes.
    // The default implemention merely prints a message.
    virtual
    void
    do_on_ssl_oob_byte(char data);

    // For a server-side socket, this callback is invoked by the network thread
    // when ALPN has been requested by the client. This function should return
    // the name of protocol being selected. If an empty string is returned, no
    // ALPN protocol will be selected. The argument is the list of protocols
    // that have been offered by the client.
    // The default implemention returns an empty string.
    virtual
    charbuf_256
    do_on_ssl_alpn_request(cow_vector<charbuf_256>&& protos);

    // For a client-side socket, this function offers a list of protocols to the
    // server. This function must be called before SSL negotiation, for example
    // inside the constructor of a derived class or just before assigning this
    // socket to the network driver. The argument is the list of protocols that
    // will be offered to the server. Empty protocol names are ignored. If the
    // list is empty, ALPN is not requested.
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
    remote_address() const;

    // Gets the protocol that has been selected by ALPN.
    // For a server-side socket, this string equals the result of a previous
    // `do_on_ssl_alpn_request()` callback. For a client-side socket, this
    // string is only available since the `do_on_ssl_connected()` callback.
    // If no ALPN protocol has been selected, an empty string is returned.
    const cow_string&
    alpn_protocol() const noexcept
      { return this->m_alpn_proto;  }

    // Enqueues some bytes for sending.
    // If this function returns `true`, data will have been enqueued; however it
    // is not guaranteed that they will arrive at the destination host. If this
    // function returns `false`, the connection will have been closed.
    // If this function throws an exception, there is no effect.
    // This function is thread-safe.
    bool
    ssl_send(const char* data, size_t size);

    bool
    ssl_send(const linear_buffer& data);

    bool
    ssl_send(const cow_string& data);

    bool
    ssl_send(const string& data);

    // Sends an out-of-band byte. OOB bytes can be sent even when there are
    // pending normal data. This function never blocks. If the OOB byte cannot
    // be sent, `false` is returned and there is no effect.
    // This function is thread-safe.
    bool
    ssl_send_oob(char data) noexcept;

    // Shuts the socket down gracefully.
    // This function is thread-safe.
    bool
    ssl_shut_down() noexcept;
  };

}  // namespace poseidon
#endif
