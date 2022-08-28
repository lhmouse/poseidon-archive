// This file is part of Poseidon.
// Copyleft 2022, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_SOCKET_ABSTRACT_SOCKET_
#define POSEIDON_SOCKET_ABSTRACT_SOCKET_

#include "../fwd.hpp"
#include "enums.hpp"
#include "socket_address.hpp"

namespace poseidon {

class Abstract_Socket
  {
  private:
    friend class Network_Driver;

    unique_posix_fd m_fd;
    atomic_relaxed<Socket_State> m_state = { socket_state_unknown };

    mutable atomic_acq_rel<bool> m_sockname_ready;
    mutable plain_mutex m_sockname_mutex;
    mutable cow_string m_sockname;

    mutable recursive_mutex m_io_mutex;
    Network_Driver* m_io_driver;
    bool m_io_throttled = false;
    linear_buffer m_io_read_queue;
    linear_buffer m_io_write_queue;

  protected:
    // Takes ownership of an existent socket.
    explicit
    Abstract_Socket(unique_posix_fd&& fd);

    // Creates a new non-blocking socket.
    explicit
    Abstract_Socket(int family, int type, int protocol);

  protected:
    // Gets the network driver instance inside the callbacks hereafter.
    // If this function is called elsewhere, the behavior is undefined.
    Network_Driver&
    do_abstract_socket_lock_driver(recursive_mutex::unique_lock& lock) const noexcept
      {
        lock.lock(this->m_io_mutex);
        ROCKET_ASSERT(this->m_io_driver);
        return *(this->m_io_driver);
      }

    // Gets the read (receive) queue.
    linear_buffer&
    do_abstract_socket_lock_read_queue(recursive_mutex::unique_lock& lock) noexcept
      {
        lock.lock(this->m_io_mutex);
        return this->m_io_read_queue;
      }

    // Gets the write (send) queue.
    linear_buffer&
    do_abstract_socket_lock_write_queue(recursive_mutex::unique_lock& lock) noexcept
      {
        lock.lock(this->m_io_mutex);
        return this->m_io_write_queue;
      }

    // Sets the socket state to `socket_state_established`. For an outgoing
    // stream socket or a datagram socket, such a state change indicates
    // success of initialization of the socket.
    bool
    do_set_established() noexcept
      {
        Socket_State cmp = socket_state_connecting;
        bool updated = this->m_state.cmpxchg(cmp, socket_state_established);
        return updated;
      }

    // This callback is invoked by the network thread when the socket has
    // been closed, and is intended to be overriden by derived classes.
    // The argument is zero for normal closure, or an error number in the
    // case of an error.
    virtual
    void
    do_abstract_socket_on_closed(int err) = 0;

    // This callback is invoked by the network thread when incoming data are
    // available, and is intended to be overriden by derived classes.
    virtual
    void
    do_abstract_socket_on_readable() = 0;

    // This callback is invoked by the network thread when outgoing data are
    // possible, and is intended to be overriden by derived classes.
    virtual
    void
    do_abstract_socket_on_writable() = 0;

  public:
    ASTERIA_NONCOPYABLE_VIRTUAL_DESTRUCTOR(Abstract_Socket);

    // Gets the file descriptor.
    int
    fd() const noexcept
      { return this->m_fd.get();  }

    // Gets the socket state.
    Socket_State
    socket_state() const noexcept
      { return this->m_state.load();  }

    // Gets the local or bound address of this socket as a human-readable
    // string. In case of errors, a string with information about the error is
    // returned instead.
    // This function is thread-safe.
    const cow_string&
    get_local_address() const;

    // Shuts the socket down without sending any protocol-specific closure
    // notifications.
    // This function is thread-safe.
    bool
    quick_shut_down() noexcept;
  };

}  // namespace poseidon

#endif
