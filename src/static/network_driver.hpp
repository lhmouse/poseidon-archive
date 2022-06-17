// This file is part of Poseidon.
// Copyleft 2022, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_STATIC_NETWORK_DRIVER_
#define POSEIDON_STATIC_NETWORK_DRIVER_

#include "../fwd.hpp"

namespace poseidon {

// This class performs network I/O operations.
// Objects of this class are recommended to be static.
class Network_Driver
  {
  private:
    unique_posix_fd m_epoll_lt;
    unique_posix_fd m_epoll_et;

    mutable plain_mutex m_conf_mutex;
    uint32_t m_event_buffer_size = 0;
    uint32_t m_throttle_size = 0;

    mutable plain_mutex m_epoll_mutex;
    unordered_map<void*, weak_ptr<Abstract_Socket>> m_epoll_sockets;

    mutable plain_mutex m_event_mutex;
    linear_buffer m_events;

  public:
    // Constructs an empty driver.
    explicit
    Network_Driver();

  public:
    ASTERIA_NONCOPYABLE_DESTRUCTOR(Network_Driver);

    // Reloads configuration from 'main.conf'.
    // If this function fails, an exception is thrown, and there is no effect.
    // This function is thread-safe.
    void
    reload(const Config_File& file);

    // Polls sockets.
    // This function should be called by the network thread repeatedly.
    void
    thread_loop();

    // Inserts a socket for polling. The network driver will hold a weak reference
    // to this socket.
    // This function is thread-safe.
    void
    insert(const shared_ptr<Abstract_Socket>& socket);
  };

}  // namespace poseidon

#endif
