// This file is part of Poseidon.
// Copyleft 2020, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_SOCKET_ABSTRACT_UDP_SOCKET_
#define POSEIDON_SOCKET_ABSTRACT_UDP_SOCKET_

#include "abstract_socket.hpp"

namespace poseidon {

class Abstract_UDP_Socket
  : public ::asteria::Rcfwd<Abstract_UDP_Socket>,
    public Abstract_Socket
  {
  private:
    // These are I/O components.
    mutable simple_mutex m_io_mutex;
    Connection_State m_connection_state = connection_state_empty;
    linear_buffer m_rqueue, m_wqueue;  // read and write queues

  protected:
    // Creates a new non-blocking socket.
    explicit
    Abstract_UDP_Socket(::sa_family_t family);

  private:
    inline
    IO_Result
    do_socket_close_unlocked() noexcept;

    // Reads some data.
    // `lock` will lock `*this` after the call.
    IO_Result
    do_socket_on_poll_read(simple_mutex::unique_lock& lock) final;

    // Returns `0` due to lack of congestion control.
    // `lock` will lock `*this` after the call, nevertheless.
    size_t
    do_write_queue_size(simple_mutex::unique_lock& lock) const final;

    // Writes some data.
    // `lock` will lock `*this` after the call.
    IO_Result
    do_socket_on_poll_write(simple_mutex::unique_lock& lock) final;

    // Notifies that this socket has been closed.
    void
    do_socket_on_poll_close(int err) final;

  protected:
    // Binds this socket to the specified address.
    void
    do_socket_bind(const Socket_Address& addr);

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
    do_socket_on_receive(const Socket_Address& addr, linear_buffer&& data)
      = 0;

    // Notifies that this socket has been fully closed.
    // The default implementation prints a message but does nothing otherwise.
    // Please mind thread safety, as this function is called by the network thread.
    virtual
    void
    do_socket_on_close(int err);

    // Enqueues a packet for writing.
    // This function returns `true` if the data have been queued, or `false` if a
    // shutdown request has been initiated.
    // This function is thread-safe.
    bool
    do_socket_send(const Socket_Address& addr, const char* data, size_t size);

    bool
    do_socket_send(const Socket_Address& addr, const cow_string& str)
      {
        return this->do_socket_send(addr, str.data(), str.size());
      }

  public:
    ASTERIA_NONCOPYABLE_DESTRUCTOR(Abstract_UDP_Socket);

    using Abstract_Socket::get_fd;
    using Abstract_Socket::kill;
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
    close() noexcept;
  };

}  // namespace poseidon

#endif
