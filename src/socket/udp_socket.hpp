// This file is part of Poseidon.
// Copyleft 2022, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_SOCKET_UDP_SOCKET_
#define POSEIDON_SOCKET_UDP_SOCKET_

#include "../fwd.hpp"
#include "abstract_socket.hpp"

namespace poseidon {

class UDP_Socket
  : public Abstract_Socket
  {
  private:
    friend class Network_Driver;

  protected:
    // Server-side constructor:
    // Creates a socket that is bound onto the given address.
    explicit
    UDP_Socket(const Socket_Address& addr);

    // Client-side constructor:
    // Creates a socket that is bound onto a random address.
    explicit
    UDP_Socket(int family);

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

    // This callback is invoked by the network thread when a port has been opened
    // and incoming packets are possible.
    // The default implemention merely prints a message.
    virtual
    void
    do_on_udp_opened();

    // This callback is invoked by the network thread when a packet has been
    // received, and is intended to be overriden by derived classes.
    // These arguments compose a complete packet. The `data` object may be reused
    // for subsequent packets.
    virtual
    void
    do_on_udp_packet(Socket_Address&& addr, linear_buffer&& data) = 0;

  public:
    ASTERIA_NONCOPYABLE_VIRTUAL_DESTRUCTOR(UDP_Socket);

    // Joins/leaves a multicast group.
    // `maddr` is the multicast group to join/leave, and must be a valid multicast
    // address. It is also used to determine which network interface to use. `ttl`
    // specifies the TTL of multicast packets. `loopback` specifies whether packets
    // should be sent back to the sender.
    // If these functions fail, an exception is thrown, and the state of this socket
    // is unspecified.
    // These functions are thread-safe.
    void
    join_multicast_group(const Socket_Address& maddr, uint8_t ttl, bool loopback);

    void
    leave_multicast_group(const Socket_Address& maddr);

    // Enqueues a packet for sending.
    // The return value merely indicates whether the attempt has succeeded. The
    // packet may or may never arrive at the destination host.
    // This function is thread-safe.
    bool
    udp_send(const Socket_Address& addr, const char* data, size_t size);

    bool
    udp_send(const Socket_Address& addr, const linear_buffer& data);

    bool
    udp_send(const Socket_Address& addr, const cow_string& data);

    bool
    udp_send(const Socket_Address& addr, const string& data);
  };

}  // namespace poseidon

#endif
