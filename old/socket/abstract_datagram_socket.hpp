// This file is part of Poseidon.
// Copyleft 2022, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_SOCKET_ABSTRACT_DATAGRAM_SOCKET_
#define POSEIDON_SOCKET_ABSTRACT_DATAGRAM_SOCKET_

#include "abstract_socket.hpp"

namespace poseidon {

class Abstract_Datagram_Socket
  : public virtual Abstract_Socket
  {
  private:
    // This class is an interface and has no data member.

  protected:
    explicit
    Abstract_Datagram_Socket() noexcept
      { }

    POSEIDON_DELETE_COPY(Abstract_Datagram_Socket);

  protected:
    // The network driver notifies incoming data via this callback.
    virtual
    IO_Result
    do_abstract_socket_on_poll_read(simple_mutex::unique_lock& lock) final;

    // The network driver notifies possibility of outgoing data via this callback.
    virtual
    IO_Result
    do_abstract_socket_on_poll_write(simple_mutex::unique_lock& lock) final;

    // When incoming data is available, this callback is invoked.
    // Please mind thread safety, as this function is called by the network thread.
    virtual
    void
    do_abstract_datagram_socket_on_receive(Socket_Address&& addr, linear_buffer&& data)
      = 0;

    // Enqueues outgoing data.
    // This functions is thread-safe.
    bool
    do_abstract_datagram_socket_send(const Socket_Address& addr, const char* data, size_t size);

  public:
    virtual
    ~Abstract_Datagram_Socket();

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
  };

}  // namespace poseidon

#endif  // POSEIDON_SOCKET_ABSTRACT_DATAGRAM_SOCKET_
