// This file is part of Poseidon.
// Copyleft 2022, LH_Mouse. All wrongs reserved.

#include "../precompiled.ipp"
#include "udp_socket.hpp"
#include "../static/async_logger.hpp"
#include "../utils.hpp"
#include <sys/socket.h>

namespace poseidon {

UDP_Socket::
UDP_Socket(int family)
  : Abstract_Socket(family, SOCK_DGRAM, IPPROTO_UDP)
  {
  }

UDP_Socket::
~UDP_Socket()
  {
  }

IO_Result
UDP_Socket::
do_abstract_socket_on_readable()
  {
    const recursive_mutex::unique_lock io_lock(this->m_io_mutex);

    // Try getting a packet from this socket.
    this->m_io_read_queue.clear();
    this->m_io_read_queue.reserve(0xFFFFU);
    size_t datalen = this->m_io_read_queue.capacity();

    Socket_Address addr;
    ::socklen_t addrlen = (::socklen_t) addr.capacity();

    ::ssize_t r;
  try_io:
    r = ::recvfrom(this->fd(), this->m_io_read_queue.mut_end(), datalen, 0, addr.mut_addr(), &addrlen);
    if(r < 0) {
      switch(errno) {
        case EINTR:
          goto try_io;

#if EWOULDBLOCK != EAGAIN
        case EAGAIN:
#endif
        case EWOULDBLOCK:
          return io_result_would_block;
      }

      POSEIDON_THROW((
          "Error reading UDP socket",
          "[`recvfrom()` failed: $3]",
          "[UDP socket `$1` (class `$2`)]"),
          this, typeid(*this), format_errno());
    }

    // Accept this packet.
    datalen = (size_t) r;
    this->m_io_read_queue.accept(datalen);
    addr.set_size(addrlen);

    this->do_on_udp_packet(::std::move(addr), ::std::move(this->m_io_read_queue));
    return io_result_partial;
  }

IO_Result
UDP_Socket::
do_abstract_socket_on_writable()
  {
    const recursive_mutex::unique_lock io_lock(this->m_io_mutex);

    if(this->m_io_write_queue.empty())
      return io_result_partial;

    Socket_Address addr;
    ::socklen_t addrlen;
    size_t datalen;

    // Get a packet from the write queue.
    // This piece of code must match `udp_send()`.
    size_t ngot = this->m_io_write_queue.getn((char*) &addrlen, sizeof(addrlen));
    ROCKET_ASSERT(ngot == sizeof(addrlen));

    ngot = this->m_io_write_queue.getn((char*) &datalen, sizeof(datalen));
    ROCKET_ASSERT(ngot == sizeof(datalen));

    ngot = this->m_io_write_queue.getn((char*) addr.mut_data(), addrlen);
    ROCKET_ASSERT(ngot == (uint32_t) addrlen);

    // In the case of other errors, data shall be removed from the write queue after
    // the attempt to send, no matter whether the operation has succeeded or not.
    ::ssize_t r;
  try_io:
    r = ::sendto(this->fd(), this->m_io_write_queue.begin(), datalen, 0, addr.addr(), addrlen);
    this->m_io_write_queue.discard(datalen);
    if(r < 0) {
      switch(errno) {
        case EINTR:
          goto try_io;

#if EWOULDBLOCK != EAGAIN
        case EAGAIN:
#endif
        case EWOULDBLOCK:
          return io_result_would_block;
      }

      POSEIDON_THROW((
          "Error writing UDP socket",
          "[`sendto()` failed: $3]",
          "[UDP socket `$1` (class `$2`)]"),
          this, typeid(*this), format_errno());
    }

    // Report that some data have been written anyway.
    return io_result_partial;
  }

void
UDP_Socket::
do_abstract_socket_on_exception(exception& stdex)
  {
    POSEIDON_LOG_WARN((
        "Ignoring exception: $3",
        "[UDP socket `$1` (class `$2`)]"),
        this, typeid(*this), stdex);
  }

void
UDP_Socket::
join_multicast_group(const Socket_Address& maddr, uint8_t ttl, bool loopback)
  {
    if(maddr.classify() != socket_address_class_multicast)
      POSEIDON_THROW((
          "Invalid multicast address `$1`"),
          maddr);

    if(maddr.family() == AF_INET) {
      // IPv4
      struct ::ip_mreqn mreq;
      mreq.imr_multiaddr = maddr.addr4()->sin_addr;
      mreq.imr_address.s_addr = INADDR_ANY;
      mreq.imr_ifindex = 0;
      if(::setsockopt(this->fd(), IPPROTO_IP, IP_ADD_MEMBERSHIP, &mreq, sizeof(mreq)) != 0)
        POSEIDON_THROW((
            "Failed to join multicast group `$4`",
            "[`setsockopt()` failed: $3]",
            "[UDP socket `$1` (class `$2`)]"),
            this, typeid(*this), format_errno(), maddr);

      int ival = ttl;
      if(::setsockopt(this->fd(), IPPROTO_IP, IP_MULTICAST_TTL, &ival, sizeof(ival)) != 0)
        POSEIDON_THROW((
            "Failed to set TTL of multicast packets",
            "[`setsockopt()` failed: $3]",
            "[UDP socket `$1` (class `$2`)]"),
            this, typeid(*this), format_errno());

      ival = loopback;
      if(::setsockopt(this->fd(), IPPROTO_IP, IP_MULTICAST_LOOP, &ival, sizeof(ival)) != 0)
        POSEIDON_THROW((
            "Failed to set loopback of multicast packets",
            "[`setsockopt()` failed: $3]",
            "[UDP socket `$1` (class `$2`)]"),
            this, typeid(*this), format_errno());
    }
    else if(maddr.family() == AF_INET6) {
      // IPv6
      struct ::ipv6_mreq mreq;
      mreq.ipv6mr_multiaddr = maddr.addr6()->sin6_addr;
      mreq.ipv6mr_interface = 0;
      if(::setsockopt(this->fd(), IPPROTO_IPV6, IPV6_ADD_MEMBERSHIP, &mreq, sizeof(mreq)) != 0)
        POSEIDON_THROW((
            "Failed to join multicast group `$4`",
            "[`setsockopt()` failed: $3]",
            "[UDP socket `$1` (class `$2`)]"),
            this, typeid(*this), format_errno(), maddr);

      int ival = ttl;
      if(::setsockopt(this->fd(), IPPROTO_IPV6, IPV6_MULTICAST_HOPS, &ival, sizeof(ival)) != 0)
        POSEIDON_THROW((
            "Failed to set TTL of multicast packets",
            "[`setsockopt()` failed: $3]",
            "[UDP socket `$1` (class `$2`)]"),
            this, typeid(*this), format_errno());

      ival = loopback;
      if(::setsockopt(this->fd(), IPPROTO_IPV6, IPV6_MULTICAST_LOOP, &ival, sizeof(ival)) != 0)
        POSEIDON_THROW((
            "Failed to set loopback of multicast packets",
            "[`setsockopt()` failed: $3]",
            "[UDP socket `$1` (class `$2`)]"),
            this, typeid(*this), format_errno());
    }
    else
      POSEIDON_THROW((
          "Socket address family `$3` not supported",
          "[UDP socket `$1` (class `$2`)]"),
          this, typeid(*this), maddr.family());
  }

void
UDP_Socket::
leave_multicast_group(const Socket_Address& maddr)
  {
    if(maddr.classify() != socket_address_class_multicast)
      POSEIDON_THROW((
          "Invalid multicast address `$1`"),
          maddr);

    if(maddr.family() == AF_INET) {
      // IPv4
      struct ::ip_mreqn mreq;
      mreq.imr_multiaddr = maddr.addr4()->sin_addr;
      mreq.imr_address.s_addr = INADDR_ANY;
      mreq.imr_ifindex = 0;
      if(::setsockopt(this->fd(), IPPROTO_IP, IP_DROP_MEMBERSHIP, &mreq, sizeof(mreq)) != 0)
        POSEIDON_THROW((
            "Failed to leave multicast group `$4`",
            "[`setsockopt()` failed: $3]",
            "[UDP socket `$1` (class `$2`)]"),
            this, typeid(*this), format_errno(), maddr);
    }
    else if(maddr.family() == AF_INET6) {
      // IPv6
      struct ::ipv6_mreq mreq;
      mreq.ipv6mr_multiaddr = maddr.addr6()->sin6_addr;
      mreq.ipv6mr_interface = 0;
      if(::setsockopt(this->fd(), IPPROTO_IPV6, IPV6_DROP_MEMBERSHIP, &mreq, sizeof(mreq)) != 0)
        POSEIDON_THROW((
            "Failed to leave multicast group `$4`",
            "[`setsockopt()` failed: $3]",
            "[UDP socket `$1` (class `$2`)]"),
            this, typeid(*this), format_errno(), maddr);
    }
    else
      POSEIDON_THROW((
          "Socket address family `$3` not supported",
          "[UDP socket `$1` (class `$2`)]"),
          this, typeid(*this), maddr.family());
  }

bool
UDP_Socket::
udp_send(const Socket_Address& addr, const char* data, size_t size)
  {
    if(size && !data)
      POSEIDON_THROW((
          "Null data pointer",
          "[UDP socket `$1` (class `$2`)]"),
          this, typeid(*this));

    size_t datalen = size & 0xFFFFU;
    if(datalen != size)
      POSEIDON_THROW((
          "`$3` bytes is too large for a UDP packet",
          "[UDP socket `$1` (class `$2`)]"),
          this, typeid(*this), size);

    // If this socket has been marked closed, fail immediately.
    if(this->socket_state() == socket_state_closed)
      return false;

    const recursive_mutex::unique_lock io_lock(this->m_io_mutex);

    // Encode this packet.
    // This piece of code must match `do_abstract_socket_on_writable()`.
    ::socklen_t addrlen = addr.ssize();
    size_t queue_size = sizeof(addrlen) + sizeof(datalen) + (uint32_t) addrlen + datalen;
    this->m_io_write_queue.reserve(queue_size);

    ::memcpy(this->m_io_write_queue.mut_end(), &addrlen, sizeof(addrlen));
    this->m_io_write_queue.accept(sizeof(addrlen));

    ::memcpy(this->m_io_write_queue.mut_end(), &datalen, sizeof(datalen));
    this->m_io_write_queue.accept(sizeof(datalen));

    ::memcpy(this->m_io_write_queue.mut_end(), addr.data(), (uint32_t) addrlen);
    this->m_io_write_queue.accept((uint32_t) addrlen);

    ::memcpy(this->m_io_write_queue.mut_end(), data, datalen);
    this->m_io_write_queue.accept(datalen);

    // Try writing once. This is essential for the edge-triggered epoll to work
    // reliably, because the level-triggered epoll does not check for `EPOLLOUT` by
    // default. If the packet has been sent anyway, discard it from the write queue.
    this->do_abstract_socket_on_writable();
    return true;
  }

bool
UDP_Socket::
udp_send(const Socket_Address& addr, const cow_string& data)
  {
    return this->udp_send(addr, data.data(), data.size());
  }

bool
UDP_Socket::
udp_send(const Socket_Address& addr, const string& data)
  {
    return this->udp_send(addr, data.data(), data.size());
  }

}  // namespace poseidon
