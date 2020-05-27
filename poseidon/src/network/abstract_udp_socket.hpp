// This file is part of Poseidon.
// Copyleft 2020, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_NETWORK_ABSTRACT_UDP_SOCKET_HPP_
#define POSEIDON_NETWORK_ABSTRACT_UDP_SOCKET_HPP_

#include "abstract_socket.hpp"
#include <rocket/linear_buffer.hpp>

namespace poseidon {

class Abstract_UDP_Socket
  : public Abstract_Socket
  {
  private:
    mutable Si_Mutex m_mutex;
    Connection_State m_cstate = connection_state_initial;
    ::rocket::linear_buffer m_wqueue;  // write queue

  public:
    explicit
    Abstract_UDP_Socket(unique_posix_fd&& fd)
      : Abstract_Socket(::std::move(fd))
      { }

    ASTERIA_NONCOPYABLE_DESTRUCTOR(Abstract_UDP_Socket);

  private:
    inline
    IO_Result
    do_async_shutdown_nolock()
    noexcept;

    // Reads some data.
    // `lock` will lock `*this` after the call.
    // `hint` is used as the I/O buffer. `size` specifies the maximum number of
    // bytes to read.
    IO_Result
    do_on_async_poll_read(Si_Mutex::unique_lock& lock, void* hint, size_t size)
    final;

    // Returns `0` due to lack of congestion control.
    // `lock` will lock `*this` after the call, nevertheless.
    size_t
    do_write_queue_size(Si_Mutex::unique_lock& lock)
    const final;

    // Writes some data.
    // `lock` will lock `*this` after the call.
    // `hint` and `size` are ignored.
    IO_Result
    do_on_async_poll_write(Si_Mutex::unique_lock& lock, void* hint, size_t size)
    final;

    // Notifies this socket has been closed.
    void
    do_on_async_poll_shutdown(int err)
    final;

  protected:
    // Notifies a socket has been open for sending data.
    // The default implementation prints a message but does nothing otherwise.
    // Please mind thread safety, as this function is called by the network thread.
    virtual
    void
    do_on_async_establish();

    // Consumes an incoming packet.
    // Please mind thread safety, as this function is called by the network thread.
    virtual
    void
    do_on_async_receive(Socket_Address&& addr, void* data, size_t size)
      = 0;

    // Notifies this socket has been fully closed.
    // The default implementation prints a message but does nothing otherwise.
    // Please mind thread safety, as this function is called by the network thread.
    virtual
    void
    do_on_async_shutdown(int err);

  public:
    using Abstract_Socket::get_fd;
    using Abstract_Socket::abort;
    using Abstract_Socket::get_local_address;

    // Binds this socket to the specified address.
    void
    bind(const Socket_Address& addr);

    // Sets multicast parameters.
    // `ifindex` is the inteface index (zero = use default).
    // `ifname` is the interface name (empty string = use default).
    // `ttl` sets the TTL of packets.
    // `loop` specifies whether packets should be looped back to the sender.
    // If this function fails, an exception is thrown, and the state of this socket
    // is unspecified.
    void
    set_multicast(int ifindex, uint8_t ttl, bool loop);

    void
    set_multicast(const char* ifname, uint8_t ttl, bool loop);

    // Joins/leaves a multicast group.
    // `maddr` is the multicast group to join/leave.
    // `ifindex` is the inteface index (zero = use default).
    // `ifname` is the interface name (empty string = use default).
    // If this function fails, an exception is thrown, and the state of this socket
    // is unspecified.
    // This function is thread-safe.
    void
    join_multicast_group(const Socket_Address& maddr, int ifindex);

    void
    join_multicast_group(const Socket_Address& maddr, const char* ifname);

    void
    leave_multicast_group(const Socket_Address& maddr, int ifindex);

    void
    leave_multicast_group(const Socket_Address& maddr, const char* ifname);

    // Enqueues a packet for writing.
    // This function returns `true` if the data have been queued, or `false` if a
    // shutdown request has been initiated.
    // This function is thread-safe.
    bool
    async_send(const Socket_Address& addr, const void* data, size_t size);

    // Marks this socket as closed immediately. No further data may be read from or
    // written through it.
    bool
    async_shutdown()
    noexcept;
  };

}  // namespace poseidon

#endif
