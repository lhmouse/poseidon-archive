// This file is part of Poseidon.
// Copyleft 2022, LH_Mouse. All wrongs reserved.

#include "../precompiled.ipp"
#include "udp_socket.hpp"
#include "../static/async_logger.hpp"
#include "../utils.hpp"
#include <sys/socket.h>
#include <sys/ioctl.h>
#include <net/if.h>

namespace poseidon {

UDP_Socket::
UDP_Socket(const Socket_Address& saddr)
  : Abstract_Socket(SOCK_DGRAM, IPPROTO_UDP)
  {
    // Use `SO_REUSEADDR`. Errors are ignored.
    int ival = 1;
    ::setsockopt(this->fd(), SOL_SOCKET, SO_REUSEADDR, &ival, sizeof(ival));

    // Bind this socket onto `addr`.
    ::sockaddr_in6 addr;
    addr.sin6_family = AF_INET6;
    addr.sin6_port = htobe16(saddr.port());
    addr.sin6_flowinfo = 0;
    addr.sin6_addr = saddr.addr();
    addr.sin6_scope_id = 0;

    if(::bind(this->fd(), (const ::sockaddr*) &addr, sizeof(addr)) != 0)
      POSEIDON_THROW((
          "Failed to bind UDP socket onto `$4`",
          "[`bind()` failed: $3]",
          "[UDP socket `$1` (class `$2`)]"),
          this, typeid(*this), format_errno(), saddr);

    POSEIDON_LOG_INFO((
        "UDP server started listening on `$3`",
        "[UDP socket `$1` (class `$2`)]"),
        this, typeid(*this), this->local_address());
  }

UDP_Socket::
UDP_Socket()
  : Abstract_Socket(SOCK_DGRAM, IPPROTO_UDP)
  {
  }

UDP_Socket::
~UDP_Socket()
  {
  }

void
UDP_Socket::
do_abstract_socket_on_closed(int err)
  {
    POSEIDON_LOG_INFO((
        "UDP socket on `$3` closed: $4",
        "[UDP socket `$1` (class `$2`)]"),
        this, typeid(*this), this->local_address(), format_errno(err));
  }

void
UDP_Socket::
do_abstract_socket_on_readable()
  {
    recursive_mutex::unique_lock io_lock;
    auto& queue = this->do_abstract_socket_lock_read_queue(io_lock);
    auto& saddr = this->m_recv_saddr;
    ::ssize_t io_result = 0;

    for(;;) {
      // Try getting a packet.
      queue.clear();
      queue.reserve(0xFFFFU);

      ::sockaddr_in6 addr;
      ::socklen_t addrlen = sizeof(addr);
      io_result = ::recvfrom(this->fd(), queue.mut_end(), queue.capacity(), 0, (::sockaddr*) &addr, &addrlen);

      if(io_result < 0) {
        if((errno == EAGAIN) || (errno == EWOULDBLOCK))
          break;

        POSEIDON_LOG_ERROR((
            "Error reading UDP socket",
            "[`recvfrom()` failed: $3]",
            "[UDP socket `$1` (class `$2`)]"),
            this, typeid(*this), format_errno());

        // Errors are ignored.
        continue;
      }

      if((addr.sin6_family != AF_INET6) || (addrlen != sizeof(addr)))
        continue;

      // Accept this incoming packet.
      saddr.set_addr(addr.sin6_addr);
      saddr.set_port(be16toh(addr.sin6_port));
      queue.accept((size_t) io_result);

      this->do_on_udp_packet(::std::move(saddr), ::std::move(queue));
    }
  }

void
UDP_Socket::
do_abstract_socket_on_oob_readable()
  {
    // UDP doesn't support OOB data.
  }

void
UDP_Socket::
do_abstract_socket_on_writable()
  {
    recursive_mutex::unique_lock io_lock;
    auto& queue = this->do_abstract_socket_lock_write_queue(io_lock);
    ::ssize_t io_result = 0;

    for(;;) {
      // Get a packet from the write queue. In case of errors, no attempt is made
      // to re-send failed packets.
      if(queue.size() == 0)
        break;

      ::sockaddr_in6 addr;
      uint16_t datalen;

      // This must match `udp_send()`.
      size_t ngot = queue.getn((char*) &addr, sizeof(addr));
      ROCKET_ASSERT(ngot == sizeof(addr));
      ngot = queue.getn((char*) &datalen, sizeof(datalen));
      ROCKET_ASSERT(ngot == sizeof(datalen));
      io_result = ::sendto(this->fd(), queue.begin(), datalen, 0, (const ::sockaddr*) &addr, sizeof(addr));
      queue.discard(datalen);

      if(io_result < 0) {
        if((errno == EAGAIN) || (errno == EWOULDBLOCK))
          break;

        POSEIDON_LOG_ERROR((
            "Error writing UDP socket",
            "[`sendto()` failed: $3]",
            "[UDP socket `$1` (class `$2`)]"),
            this, typeid(*this), format_errno());

        // Errors are ignored.
        continue;
      }

      ROCKET_ASSERT((size_t) io_result <= datalen);
    }

    if(this->do_abstract_socket_set_state(socket_state_connecting, socket_state_established)) {
      // Deliver the establishment notification.
      POSEIDON_LOG_DEBUG(("UDP port opened: local = $1"), this->local_address());
      this->do_on_udp_opened();
    }
  }

void
UDP_Socket::
do_on_udp_opened()
  {
    POSEIDON_LOG_INFO((
        "UDP socket on `$3` opened",
        "[UDP socket `$1` (class `$2`)]"),
        this, typeid(*this), this->local_address());
  }

void
UDP_Socket::
join_multicast_group(const Socket_Address& maddr, uint8_t ttl, bool loopback, const char* ifname_opt)
  {
    // Get the local address of the interface to use.
    struct ::ifreq ifreq;
    ::memset(&ifreq, 0, sizeof(ifreq));

    if(ifname_opt) {
      size_t ifname_len = ::strlen(ifname_opt);
      if(ifname_len + 1 > IF_NAMESIZE)
        POSEIDON_THROW((
            "Network interface name too long: ifname = `$3`",
            "[UDP socket `$1` (class `$2`)]"),
            this, typeid(*this), ifname_opt);

      ::memcpy(ifreq.ifr_name, ifname_opt, ifname_len);
    }

    // IPv6 doesn't take IPv4-mapped multicast addresses, so there has to be
    // special treatement. `sendto()` is not affected.
    if(::memcmp(maddr.data(), "\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xFF\xFF", 12) == 0) {
      // IPv4
      if(ifreq.ifr_name[0] && (::ioctl(this->fd(), (int) SIOCGIFADDR, &ifreq) != 0))
        POSEIDON_THROW((
            "Failed to get address of interface `$4`",
            "[`ioctl()` failed: $3]",
            "[UDP socket `$1` (class `$2`)]"),
            this, typeid(*this), format_errno(), ifreq.ifr_name);

      // Join the multicast group.
      struct ::ip_mreq mreq;
      ::memcpy(&(mreq.imr_multiaddr.s_addr), maddr.data() + 12, 4);
      ::memcpy(&(mreq.imr_interface.s_addr), &(ifreq.ifr_addr), 4);
      if(::setsockopt(this->fd(), IPPROTO_IP, IP_ADD_MEMBERSHIP, &mreq, sizeof(mreq)) != 0)
        POSEIDON_THROW((
            "Failed to join IPv4 multicast group `$4`",
            "[`setsockopt()` failed: $3]",
            "[UDP socket `$1` (class `$2`)]"),
            this, typeid(*this), format_errno(), maddr);

      // Set multicast parameters.
      int ival = ttl;
      if(::setsockopt(this->fd(), IPPROTO_IP, IP_MULTICAST_TTL, &ival, sizeof(ival)) != 0)
        POSEIDON_THROW((
            "Failed to set TTL of IPv4 multicast packets",
            "[`setsockopt()` failed: $3]",
            "[UDP socket `$1` (class `$2`)]"),
            this, typeid(*this), format_errno());

      ival = loopback;
      if(::setsockopt(this->fd(), IPPROTO_IP, IP_MULTICAST_LOOP, &ival, sizeof(ival)) != 0)
        POSEIDON_THROW((
            "Failed to set loopback of IPv4 multicast packets",
            "[`setsockopt()` failed: $3]",
            "[UDP socket `$1` (class `$2`)]"),
            this, typeid(*this), format_errno());
    }
    else {
      // IPv6
      if(ifreq.ifr_name[0] && (::ioctl(this->fd(), (int) SIOCGIFINDEX, &ifreq) != 0))
        POSEIDON_THROW((
            "Failed to get index of interface `$4`",
            "[`ioctl()` failed: $3]",
            "[UDP socket `$1` (class `$2`)]"),
            this, typeid(*this), format_errno(), ifreq.ifr_name);

      // Join the multicast group.
      struct ::ipv6_mreq mreq;
      mreq.ipv6mr_multiaddr = maddr.addr();
      mreq.ipv6mr_interface = (unsigned) ifreq.ifr_ifindex;
      if(::setsockopt(this->fd(), IPPROTO_IPV6, IPV6_ADD_MEMBERSHIP, &mreq, sizeof(mreq)) != 0)
        POSEIDON_THROW((
            "Failed to join IPv6 multicast group `$4`",
            "[`setsockopt()` failed: $3]",
            "[UDP socket `$1` (class `$2`)]"),
            this, typeid(*this), format_errno(), maddr);

      // Set multicast parameters.
      int ival = ttl;
      if(::setsockopt(this->fd(), IPPROTO_IPV6, IPV6_MULTICAST_HOPS, &ival, sizeof(ival)) != 0)
        POSEIDON_THROW((
            "Failed to set TTL of IPv6 multicast packets",
            "[`setsockopt()` failed: $3]",
            "[UDP socket `$1` (class `$2`)]"),
            this, typeid(*this), format_errno());

      ival = loopback;
      if(::setsockopt(this->fd(), IPPROTO_IPV6, IPV6_MULTICAST_LOOP, &ival, sizeof(ival)) != 0)
        POSEIDON_THROW((
            "Failed to set loopback of IPv6 multicast packets",
            "[`setsockopt()` failed: $3]",
            "[UDP socket `$1` (class `$2`)]"),
            this, typeid(*this), format_errno());
    }

    POSEIDON_LOG_INFO((
        "UDP socket has joined multicast group: address = `$3`, interface = `$4`",
        "[UDP socket `$1` (class `$2`)]"),
        this, typeid(*this), maddr, ifname_opt);
  }

void
UDP_Socket::
leave_multicast_group(const Socket_Address& maddr, const char* ifname_opt)
  {
    // Get the local address of the interface to use.
    struct ::ifreq ifreq;
    ::memset(&ifreq, 0, sizeof(ifreq));

    if(ifname_opt) {
      size_t ifname_len = ::strlen(ifname_opt);
      if(ifname_len + 1 > IF_NAMESIZE)
        POSEIDON_THROW((
            "Network interface name too long: ifname = `$3`",
            "[UDP socket `$1` (class `$2`)]"),
            this, typeid(*this), ifname_opt);

      ::memcpy(ifreq.ifr_name, ifname_opt, ifname_len);
    }

    // IPv6 doesn't take IPv4-mapped multicast addresses, so there has to be
    // special treatement. `sendto()` is not affected.
    if(::memcmp(maddr.data(), "\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xFF\xFF", 12) == 0) {
      // IPv4
      if(ifreq.ifr_name[0] && (::ioctl(this->fd(), (int) SIOCGIFADDR, &ifreq) != 0))
        POSEIDON_THROW((
            "Failed to get address of interface `$4`",
            "[`ioctl()` failed: $3]",
            "[UDP socket `$1` (class `$2`)]"),
            this, typeid(*this), format_errno(), ifreq.ifr_name);

      // Leave the multicast group.
      struct ::ip_mreq mreq;
      ::memcpy(&(mreq.imr_multiaddr.s_addr), maddr.data() + 12, 4);
      ::memcpy(&(mreq.imr_interface.s_addr), &(ifreq.ifr_addr), 4);
      if(::setsockopt(this->fd(), IPPROTO_IP, IP_DROP_MEMBERSHIP, &mreq, sizeof(mreq)) != 0)
        POSEIDON_THROW((
            "Failed to leave IPv4 multicast group `$4`",
            "[`setsockopt()` failed: $3]",
            "[UDP socket `$1` (class `$2`)]"),
            this, typeid(*this), format_errno(), maddr);
    }
    else {
      // IPv6
      if(ifreq.ifr_name[0] && (::ioctl(this->fd(), (int) SIOCGIFINDEX, &ifreq) != 0))
        POSEIDON_THROW((
            "Failed to get index of interface `$4`",
            "[`ioctl()` failed: $3]",
            "[UDP socket `$1` (class `$2`)]"),
            this, typeid(*this), format_errno(), ifreq.ifr_name);

      // Leave the multicast group.
      struct ::ipv6_mreq mreq;
      mreq.ipv6mr_multiaddr = maddr.addr();
      mreq.ipv6mr_interface = (unsigned) ifreq.ifr_ifindex;
      if(::setsockopt(this->fd(), IPPROTO_IPV6, IPV6_DROP_MEMBERSHIP, &mreq, sizeof(mreq)) != 0)
        POSEIDON_THROW((
            "Failed to leave IPv6 multicast group `$4`",
            "[`setsockopt()` failed: $3]",
            "[UDP socket `$1` (class `$2`)]"),
            this, typeid(*this), format_errno(), maddr);
    }

    POSEIDON_LOG_INFO((
        "UDP socket has left multicast group: address = `$3`, interface = `$4`",
        "[UDP socket `$1` (class `$2`)]"),
        this, typeid(*this), maddr, ifname_opt);
  }

bool
UDP_Socket::
udp_send(const Socket_Address& saddr, const char* data, size_t size)
  {
    if((data == nullptr) && (size != 0))
      POSEIDON_THROW((
          "Null data pointer",
          "[UDP socket `$1` (class `$2`)]"),
          this, typeid(*this));

    uint16_t datalen = (uint16_t) size;
    if(datalen != size)
      POSEIDON_THROW((
          "`$3` bytes is too large for a UDP packet",
          "[UDP socket `$1` (class `$2`)]"),
          this, typeid(*this), size);

    // If this socket has been marked closed, fail immediately.
    if(this->socket_state() == socket_state_closed)
      return false;

    recursive_mutex::unique_lock io_lock;
    auto& queue = this->do_abstract_socket_lock_write_queue(io_lock);
    ::ssize_t io_result = 0;

    // Try sending the packet immediately.
    // This is valid because UDP packets can be transmitted out of order.
    ::sockaddr_in6 addr;
    addr.sin6_family = AF_INET6;
    addr.sin6_port = htobe16(saddr.port());
    addr.sin6_flowinfo = 0;
    addr.sin6_addr = saddr.addr();
    addr.sin6_scope_id = 0;

    io_result = ::sendto(this->fd(), data, size, 0, (const ::sockaddr*) &addr, sizeof(addr));

    if(io_result < 0) {
      if((errno == EAGAIN) || (errno == EWOULDBLOCK)) {
        // Buffer them until the next `do_abstract_socket_on_writable()`.
        queue.reserve(sizeof(addr) + sizeof(datalen) + datalen);

        // This must match `do_abstract_socket_on_writable()`.
        ::memcpy(queue.mut_end(), &addr, sizeof(addr));
        queue.accept(sizeof(addr));
        ::memcpy(queue.mut_end(), &datalen, sizeof(datalen));
        queue.accept(sizeof(datalen));
        ::memcpy(queue.mut_end(), data, datalen);
        queue.accept(datalen);
        return true;
      }

      POSEIDON_LOG_ERROR((
          "Error writing UDP socket",
          "[`sendto()` failed: $3]",
          "[UDP socket `$1` (class `$2`)]"),
          this, typeid(*this), format_errno());

      // Errors are ignored.
      return false;
    }

    // Partial writes are accepted without errors.
    return true;
  }

bool
UDP_Socket::
udp_send(const Socket_Address& saddr, const linear_buffer& data)
  {
    return this->udp_send(saddr, data.data(), data.size());
  }

bool
UDP_Socket::
udp_send(const Socket_Address& saddr, const cow_string& data)
  {
    return this->udp_send(saddr, data.data(), data.size());
  }

bool
UDP_Socket::
udp_send(const Socket_Address& saddr, const string& data)
  {
    return this->udp_send(saddr, data.data(), data.size());
  }

}  // namespace poseidon
