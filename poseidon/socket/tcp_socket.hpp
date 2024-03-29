// This file is part of Poseidon.
// Copyleft 2022 - 2023, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_SOCKET_TCP_SOCKET_
#define POSEIDON_SOCKET_TCP_SOCKET_

#include "../fwd.hpp"
#include "abstract_socket.hpp"

namespace poseidon {

class TCP_Socket
  : public Abstract_Socket
  {
  private:
    friend class Network_Driver;

    mutable atomic_acq_rel<bool> m_peername_ready;
    mutable Socket_Address m_peername;

  protected:
    // Server-side constructor:
    // Takes ownership of an accepted socket.
    explicit
    TCP_Socket(unique_posix_fd&& fd);

    // Client-side constructor:
    // Creates a new non-blocking socket to the target host.
    explicit
    TCP_Socket(const Socket_Address& saddr);

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
    do_on_tcp_connected();

    // This callback is invoked by the network thread when some bytes have been
    // received, and is intended to be overriden by derived classes.
    // The argument contains all data that have been accumulated so far. Callees
    // should remove processed bytes.
    virtual
    void
    do_on_tcp_stream(linear_buffer& data) = 0;

    // This callback is invoked by the network thread when an out-of-band byte
    // has been received, and is intended to be overriden by derived classes.
    // The default implemention merely prints a message.
    virtual
    void
    do_on_tcp_oob_byte(char data);

  public:
    ASTERIA_NONCOPYABLE_VIRTUAL_DESTRUCTOR(TCP_Socket);

    // Gets the remote or connected address of this socket. In case of errors,
    // `ipv6_unspecified` is returned. The result is cached and will not
    // reflect changes that other APIs may have made.
    ROCKET_PURE
    const Socket_Address&
    remote_address() const noexcept;

    // Enqueues some bytes for sending.
    // If this function returns `true`, data will have been enqueued; however it
    // is not guaranteed that they will arrive at the destination host. If this
    // function returns `false`, the connection will have been closed.
    // If this function throws an exception, there is no effect.
    // This function is thread-safe.
    bool
    tcp_send(const char* data, size_t size);

    bool
    tcp_send(const linear_buffer& data);

    bool
    tcp_send(const cow_string& data);

    bool
    tcp_send(const string& data);

    // Sends an out-of-band byte. OOB bytes can be sent even when there are
    // pending normal data. This function never blocks. If the OOB byte cannot
    // be sent, `false` is returned and there is no effect.
    // This function is thread-safe.
    bool
    tcp_send_oob(char data) noexcept;

    // Shuts the socket down gracefully.
    // This function is thread-safe.
    bool
    tcp_shut_down() noexcept;
  };

}  // namespace poseidon
#endif
