// This file is part of Poseidon.
// Copyleft 2020, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_NETWORK_ABSTRACT_SOCKET_HPP_
#define POSEIDON_NETWORK_ABSTRACT_SOCKET_HPP_

#include "../fwd.hpp"
#include "enums.hpp"

namespace poseidon {

class Abstract_Socket
  : public virtual ::asteria::Rcbase
  {
    friend Network_Driver;

  private:
    unique_posix_fd m_fd;

    // These are used by network driver.
    uint64_t m_epoll_data = 123456789;
    uint32_t m_epoll_events = 0;

  public:
    explicit
    Abstract_Socket(unique_posix_fd&& fd)
      : m_fd(::std::move(fd))
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
    do_on_async_poll_read(Si_Mutex::unique_lock& lock, void* hint, size_t size)
      = 0;

    // This function shall return the number of bytes that are pending for writing.
    // `lock` shall lock `*this` after the call if locking is supported.
    virtual
    size_t
    do_write_queue_size(Si_Mutex::unique_lock& lock)
    const
      = 0;

    // The network driver notifies possibility of outgoing data via this callback.
    // `lock` shall lock `*this` after the call if locking is supported.
    // `hint` points to a temporary buffer of `size` bytes that may be used by this
    // function for any purpose.
    // Please mind thread safety, as this function is called by the network thread.
    virtual
    IO_Result
    do_on_async_poll_write(Si_Mutex::unique_lock& lock, void* hint, size_t size)
      = 0;

    // The network driver notifies closure via this callback.
    // `err` is zero for graceful shutdown.
    // Please mind thread safety, as this function is called by the network thread.
    virtual
    void
    do_on_async_poll_shutdown(int err)
      = 0;

  public:
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
    terminate()
    noexcept;

    // Gets the (bound) address of the local peer.
    Socket_Address
    get_local_address()
    const;

    // Gets the (connected) address of the remote peer.
    // This function throws an exception if no peer has connected.
    Socket_Address
    get_remote_address()
    const;
  };

}  // namespace poseidon

#endif
