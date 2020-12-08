// This file is part of Poseidon.
// Copyleft 2020, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_SOCKET_ABSTRACT_UDP_SOCKET_HPP_
#define POSEIDON_SOCKET_ABSTRACT_UDP_SOCKET_HPP_

#include "abstract_socket.hpp"

namespace poseidon {

class Abstract_UDP_Socket
  : public ::asteria::Rcfwd<Abstract_UDP_Socket>,
    public Abstract_Socket
  {
  private:
    // These are I/O components.
    mutable simple_mutex m_io_mutex;
    Connection_State m_cstate = connection_state_empty;
    linear_buffer m_wqueue;  // write queue

  protected:
    explicit
    Abstract_UDP_Socket(unique_FD&& fd)
      : Abstract_Socket(::std::move(fd))
      { }

  private:
    inline
    IO_Result
    do_socket_close_unlocked()
      noexcept;

    // Reads some data.
    // `lock` will lock `*this` after the call.
    // `hint` is used as the I/O buffer. `size` specifies the maximum number of
    // bytes to read.
    IO_Result
    do_socket_on_poll_read(simple_mutex::unique_lock& lock, char* hint, size_t size)
      final;

    // Returns `0` due to lack of congestion control.
    // `lock` will lock `*this` after the call, nevertheless.
    size_t
    do_write_queue_size(simple_mutex::unique_lock& lock)
      const final;

    // Writes some data.
    // `lock` will lock `*this` after the call.
    // `hint` and `size` are ignored.
    IO_Result
    do_socket_on_poll_write(simple_mutex::unique_lock& lock, char* hint, size_t size)
      final;

    // Notifies that this socket has been closed.
    void
    do_socket_on_poll_close(int err)
      final;

  protected:
    // Notifies that this socket has been open for incoming data.
    // The default implementation prints a message but does nothing otherwise.
    // Please mind thread safety, as this function is called by the network thread.
    virtual
    void
    do_socket_on_establish();

    // Consumes an incoming packet.
    // Please mind thread safety, as this function is called by the network thread.
    virtual
    void
    do_socket_on_receive(Socket_Address&& addr, char* data, size_t size)
      = 0;

    // Notifies that this socket has been fully closed.
    // The default implementation prints a message but does nothing otherwise.
    // Please mind thread safety, as this function is called by the network thread.
    virtual
    void
    do_socket_on_close(int err);

    // Binds this socket to the specified address.
    void
    do_bind(const Socket_Address& addr);

    // Enqueues a packet for writing.
    // This function returns `true` if the data have been queued, or `false` if a
    // shutdown request has been initiated.
    // This function is thread-safe.
    bool
    do_socket_send(const Socket_Address& addr, const char* data, size_t size);

  public:
    ASTERIA_NONCOPYABLE_DESTRUCTOR(Abstract_UDP_Socket);

    using Abstract_Socket::get_fd;
    using Abstract_Socket::terminate;
    using Abstract_Socket::get_local_address;

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

    // Marks this socket as closed immediately. No further data may be read from or
    // written through it.
    bool
    close()
      noexcept;
  };

}  // namespace poseidon

#endif
