// This file is part of Poseidon.
// Copyleft 2022, LH_Mouse. All wrongs reserved.

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
    mutable plain_mutex m_peername_mutex;
    mutable cow_string m_peername;

  protected:
    // Server-side constructor:
    // Takes ownership of an accepted socket.
    explicit
    TCP_Socket(unique_posix_fd&& fd);

    // Client-side constructor:
    // Creates a new non-blocking socket to the target host.
    explicit
    TCP_Socket(const Socket_Address& addr);

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

  public:
    ASTERIA_NONCOPYABLE_VIRTUAL_DESTRUCTOR(TCP_Socket);

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
    tcp_send(const char* data, size_t size);

    bool
    tcp_send(const linear_buffer& data);

    bool
    tcp_send(const cow_string& data);

    bool
    tcp_send(const string& data);

    // Shuts the socket down gracefully.
    // This function is thread-safe.
    bool
    tcp_shut_down() noexcept;
  };

}  // namespace poseidon

#endif
