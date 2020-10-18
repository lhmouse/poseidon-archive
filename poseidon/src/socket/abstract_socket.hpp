// This file is part of Poseidon.
// Copyleft 2020, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_SOCKET_ABSTRACT_SOCKET_HPP_
#define POSEIDON_SOCKET_ABSTRACT_SOCKET_HPP_

#include "../fwd.hpp"
#include "enums.hpp"
#include "socket_address.hpp"

namespace poseidon {

class Abstract_Socket
  : public ::asteria::Rcfwd<Abstract_Socket>
  {
    friend Network_Driver;

  private:
    atomic_relaxed<bool> m_resident;  // don't delete if orphaned
    unique_FD m_fd;

    // These are used by network driver.
    uint64_t m_epoll_data = 1234567890123456789;
    uint32_t m_epoll_events = UINT32_MAX;

    // This the local address. It is initialized upon the first request.
    mutable once_flag m_local_addr_once;
    mutable Socket_Address m_local_addr;

  public:
    explicit
    Abstract_Socket(unique_FD&& fd)
      : m_resident(false), m_fd(::std::move(fd))
      { this->do_set_common_options();  }

    ASTERIA_NONCOPYABLE_DESTRUCTOR(Abstract_Socket);

  private:
    // Enables non-blocking mode, etc.
    void
    do_set_common_options();

  protected:
    // The network driver notifies incoming data via this callback.
    // `lock` shall lock `*this` after the call if locking is supported.
    // `hint` points to a temporary buffer of `size` bytes that may be used by this
    // function for any purpose.
    // Please mind thread safety, as this function is called by the network thread.
    virtual
    IO_Result
    do_on_async_poll_read(simple_mutex::unique_lock& lock, void* hint, size_t size)
      = 0;

    // This function shall return the number of bytes that are pending for writing.
    // `lock` shall lock `*this` after the call if locking is supported.
    virtual
    size_t
    do_write_queue_size(simple_mutex::unique_lock& lock)
      const
      = 0;

    // The network driver notifies possibility of outgoing data via this callback.
    // `lock` shall lock `*this` after the call if locking is supported.
    // `hint` points to a temporary buffer of `size` bytes that may be used by this
    // function for any purpose.
    // Please mind thread safety, as this function is called by the network thread.
    virtual
    IO_Result
    do_on_async_poll_write(simple_mutex::unique_lock& lock, void* hint, size_t size)
      = 0;

    // The network driver notifies closure via this callback.
    // `err` is zero for graceful shutdown.
    // Please mind thread safety, as this function is called by the network thread.
    virtual
    void
    do_on_async_poll_shutdown(int err)
      = 0;

  public:
    // Marks this socket to be deleted if network driver holds its last reference.
    bool
    set_resident(bool value = true)
      noexcept
      { return this->m_resident.exchange(value);  }

    // Returns the stream descriptor.
    // This is used to query and adjust stream flags. You shall not perform I/O
    // operations on it.
    ROCKET_PURE_FUNCTION
    int
    get_fd()
      const noexcept
      { return this->m_fd;  }

    // Causes abnormal termination of this stream.
    // Any data that have not been delivered to the other peer are discarded.
    // The other peer is likely to see a 'connection reset by peer' error.
    void
    abort()
      noexcept;

    // Gets the (bound) address of the local peer.
    const Socket_Address&
    get_local_address()
      const;
  };

}  // namespace poseidon

#endif
