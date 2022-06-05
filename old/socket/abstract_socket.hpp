// This file is part of Poseidon.
// Copyleft 2020, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_SOCKET_ABSTRACT_SOCKET_
#define POSEIDON_SOCKET_ABSTRACT_SOCKET_

#include "../fwd.hpp"
#include "enums.hpp"
#include "socket_address.hpp"

namespace poseidon {

class Abstract_Socket
  {
    friend Network_Driver;

  private:
    unique_FD m_fd;

    // These are used by network driver.
    uint64_t m_epoll_data;
    uint32_t m_epoll_events;

    // This the local address. It is initialized upon the first request.
    mutable once_flag m_local_addr_once;
    mutable Socket_Address m_local_addr;

  protected:
    // These are I/O components.
    mutable simple_mutex m_io_mutex;
    Connection_State m_conn_state = connection_state_empty;
    linear_buffer m_queue_recv;
    linear_buffer m_queue_send;

  protected:
    // Creates a new non-blocking socket.
    explicit
    Abstract_Socket(::sa_family_t family, int type, int protocol);

    // Adopts a foreign or accepted socket.
    explicit
    Abstract_Socket(unique_FD&& fd);

    POSEIDON_DELETE_COPY(Abstract_Socket);

  private:
    // The network driver notifies incoming data via this callback.
    // `lock` should lock `*this` after the call if locking is supported.
    // Please mind thread safety, as this function is called by the network thread.
    virtual
    IO_Result
    do_abstract_socket_on_poll_read(simple_mutex::unique_lock& lock)
      = 0;

    // The network driver notifies possibility of outgoing data via this callback.
    // `lock` should lock `*this` after the call if locking is supported.
    // Please mind thread safety, as this function is called by the network thread.
    virtual
    IO_Result
    do_abstract_socket_on_poll_write(simple_mutex::unique_lock& lock)
      = 0;

    // The network driver notifies closure via this callback. This function has a
    // default implementation that prints a message.
    // `err` is zero for graceful shutdown, or a system error number otherwise.
    // Please mind thread safety, as this function is called by the network thread.
    virtual
    void
    do_abstract_socket_on_poll_close(int err);

  public:
    virtual
    ~Abstract_Socket();

    // Returns the stream descriptor.
    // This is used to query and adjust stream flags. You shall not perform I/O
    // operations on it.
    ROCKET_PURE
    int
    get_fd() const noexcept
      { return this->m_fd;  }

    // Causes abnormal termination of this stream.
    // Any data that have not been delivered to the other peer are discarded.
    // The other peer is likely to see a 'connection reset by peer' error.
    void
    kill() noexcept;

    // Gets the (bound) address of the local peer.
    const Socket_Address&
    get_local_address() const;
  };

}  // namespace poseidon

#endif
