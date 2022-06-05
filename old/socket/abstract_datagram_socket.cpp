// This file is part of Poseidon.
// Copyleft 2022, LH_Mouse. All wrongs reserved.

#include "../precompiled.ipp"
#include "abstract_datagram_socket.hpp"
#include "../static/async_logger.hpp"
#include "../static/network_driver.hpp"
#include "../utils.hpp"
#include <net/if.h>

namespace poseidon {
namespace {

int
do_ifname_to_ifindex(const char* ifname)
  {
    if(!ifname || !*ifname)
      return 0;  // use default

    auto ifindex = ::if_nametoindex(ifname);
    if(ifindex == 0)
      POSEIDON_THROW(
          "invalid network interface `$2`\n"
          "[`if_nametoindex()` failed: $1]",
          format_errno(), ifname);

    return (int) ifindex;
  }

struct Packet_Header
  {
    size_t addrlen;
    size_t datalen;
  };

}  // namespace

IO_Result
Abstract_Datagram_Socket::
do_abstract_socket_on_poll_read(simple_mutex::unique_lock& lock)
  {
    lock.lock(this->m_io_mutex);

    try {
      // Read a packet.
      this->m_queue_recv.clear();
      this->m_queue_recv.reserve(65536);
      Socket_Address::storage addrst;
      ::socklen_t addrlen = sizeof(addrst);

      ::ssize_t nrecv = ::recvfrom(this->get_fd(), this->m_queue_recv.mut_end(), this->m_queue_recv.capacity(), 0, &addrst.addr, &addrlen);
      if(nrecv < 0)
        return get_io_result_from_errno("recvfrom", errno);

      this->m_queue_recv.accept((size_t) nrecv);
      lock.unlock();

      // Process it.
      this->do_abstract_datagram_socket_on_receive(Socket_Address(addrst, addrlen), ::std::move(this->m_queue_recv));
    }
    catch(exception& stdex) {
      POSEIDON_LOG_ERROR(
          "Error processing incoming data: $1\n"
          "[socket class `$2`]",
          stdex, typeid(*this));
    }

    lock.lock(this->m_io_mutex);
    return io_result_partial_work;
  }

IO_Result
Abstract_Datagram_Socket::
do_abstract_socket_on_poll_write(simple_mutex::unique_lock& lock)
  {
    lock.lock(this->m_io_mutex);

    // Get a pending packet.
    Packet_Header header;
    size_t size = this->m_queue_send.getn((char*) &header, sizeof(header));
    if(size == 0)
      return io_result_end_of_stream;

    ROCKET_ASSERT(size == sizeof(header));
    ROCKET_ASSERT(this->m_queue_send.size() >= header.addrlen + header.datalen);

    Socket_Address::storage addrst;
    size = this->m_queue_send.getn((char*) &addrst, header.addrlen);
    ROCKET_ASSERT(size == header.addrlen);

    // Send the payload. No matter whether the operation succeeds or not, it shall
    // be removed from the send queue. Errors are ignored.
    (void)! ::sendto(this->get_fd(), this->m_queue_send.data(), header.datalen, 0, &addrst.addr, (::socklen_t) header.addrlen);
    this->m_queue_send.discard(header.datalen);

    lock.lock(this->m_io_mutex);
    return io_result_partial_work;
  }

bool
Abstract_Datagram_Socket::
do_abstract_datagram_socket_send(const Socket_Address& addr, const char* data, size_t size)
  {
    if(size > INT_MAX - sizeof(Socket_Address))
      POSEIDON_THROW("packet too large: size = $1", size);

    // Write the header.
    Packet_Header header;
    header.addrlen = addr.size();
    header.datalen = size;

    // Enqueue data. This operation must be atomic.
    simple_mutex::unique_lock lock(this->m_io_mutex);
    this->m_queue_send.reserve(sizeof(header) + header.addrlen + header.datalen);
    this->m_queue_send.putn((const char*) &header, sizeof(header));
    this->m_queue_send.putn((const char*) &(addr.data()), header.addrlen);
    this->m_queue_send.putn(data, header.datalen);
    return true;
  }

Abstract_Datagram_Socket::
~Abstract_Datagram_Socket()
  {
  }

void
Abstract_Datagram_Socket::
set_multicast(int ifindex, uint8_t ttl, bool loop)
  {
    // Get the socket family.
    ::sa_family_t family;
    ::socklen_t addrlen = sizeof(family);
    if(::getsockname(this->get_fd(), (::sockaddr*) &family, &addrlen) != 0)
      POSEIDON_THROW(
          "failed to determine socket family\n"
          "[`getsockname()` failed: $1]",
          format_errno());

    if(family == AF_INET) {
      // IPv4
      ::ip_mreqn mreq;
      mreq.imr_multiaddr.s_addr = INADDR_ANY;
      mreq.imr_address.s_addr = INADDR_ANY;
      mreq.imr_ifindex = ifindex;
      if(::setsockopt(this->get_fd(), IPPROTO_IP, IP_MULTICAST_IF, &mreq, sizeof(mreq)) != 0)
        POSEIDON_THROW(
            "could not set IPv4 multicast interface to `$2`\n"
            "[`setsockopt()` failed: $1]",
            format_errno(), ifindex);

      int value = ttl;
      if(::setsockopt(this->get_fd(), IPPROTO_IP, IP_MULTICAST_TTL, &value, sizeof(value)) != 0)
        POSEIDON_THROW(
            "could not set IPv4 multicast TTL to `$2`\n"
            "[`setsockopt()` failed: $1]",
            format_errno(), value);

      value = -loop;
      if(::setsockopt(this->get_fd(), IPPROTO_IP, IP_MULTICAST_LOOP, &value, sizeof(value)) != 0)
        POSEIDON_THROW(
            "could not set IPv4 multicast loopback to `$2`\n"
            "[`setsockopt()` failed: $1]",
            format_errno(), value);
    }
    else if(family == AF_INET6) {
      // IPv6
      if(::setsockopt(this->get_fd(), IPPROTO_IPV6, IPV6_MULTICAST_IF, &ifindex, sizeof(ifindex)) != 0)
        POSEIDON_THROW(
            "could not set IPv6 multicast interface to `$2`\n"
            "[`setsockopt()` failed: $1]",
            format_errno(), ifindex);

      int value = ttl;
      if(::setsockopt(this->get_fd(), IPPROTO_IPV6, IPV6_MULTICAST_HOPS, &value, sizeof(value)) != 0)
        POSEIDON_THROW(
            "could not set IPv6 multicast TTL to `$2`\n"
            "[`setsockopt()` failed: $1]",
            format_errno(), value);

      value = -loop;
      if(::setsockopt(this->get_fd(), IPPROTO_IPV6, IPV6_MULTICAST_LOOP, &value, sizeof(value)) != 0)
        POSEIDON_THROW(
            "could not set IPv6 multicast loopback to `$2`\n"
            "[`setsockopt()` failed: $1]",
            format_errno(), value);
    }
    else
      POSEIDON_THROW("unsupported address family `$1`", family);
  }

void
Abstract_Datagram_Socket::
set_multicast(const char* ifname, uint8_t ttl, bool loop)
  {
    return this->set_multicast(do_ifname_to_ifindex(ifname), ttl, loop);
  }

void
Abstract_Datagram_Socket::
join_multicast_group(const Socket_Address& maddr, int ifindex)
  {
    if(!maddr.is_multicast())
      POSEIDON_THROW("invalid multicast address `$1`", maddr);

    if(maddr.family() == AF_INET) {
      // IPv4
      ::ip_mreqn mreq;
      mreq.imr_multiaddr = maddr.data().addr4.sin_addr;
      mreq.imr_address.s_addr = INADDR_ANY;
      mreq.imr_ifindex = ifindex;
      if(::setsockopt(this->get_fd(), IPPROTO_IP, IP_ADD_MEMBERSHIP, &mreq, sizeof(mreq)) != 0)
        POSEIDON_THROW(
            "could not join IPv4 multicast group '$2' via interface `$3`\n"
            "[`setsockopt()` failed: $1]",
            format_errno(), maddr, ifindex);
    }
    else if(maddr.family() == AF_INET6) {
      // IPv6
      ::ipv6_mreq mreq;
      mreq.ipv6mr_multiaddr = maddr.data().addr6.sin6_addr;
      mreq.ipv6mr_interface = static_cast<unsigned>(ifindex);
      if(::setsockopt(this->get_fd(), IPPROTO_IPV6, IPV6_ADD_MEMBERSHIP, &mreq, sizeof(mreq)) != 0)
        POSEIDON_THROW(
            "could not join IPv6 multicast group '$2' via interface `$3`\n"
            "[`setsockopt()` failed: $1]",
            format_errno(), maddr, ifindex);
    }
    else
      POSEIDON_THROW("unsupported multicast address family `$1`", maddr.family());
  }

void
Abstract_Datagram_Socket::
join_multicast_group(const Socket_Address& maddr, const char* ifname)
  {
    return this->join_multicast_group(maddr, do_ifname_to_ifindex(ifname));
  }

void
Abstract_Datagram_Socket::
leave_multicast_group(const Socket_Address& maddr, int ifindex)
  {
    if(!maddr.is_multicast())
      POSEIDON_THROW("invalid multicast address `$1`", maddr);

    if(maddr.family() == AF_INET) {
      // IPv4
      ::ip_mreqn mreq;
      mreq.imr_multiaddr = maddr.data().addr4.sin_addr;
      mreq.imr_address.s_addr = INADDR_ANY;
      mreq.imr_ifindex = ifindex;
      if(::setsockopt(this->get_fd(), IPPROTO_IP, IP_DROP_MEMBERSHIP, &mreq, sizeof(mreq)) != 0)
        POSEIDON_THROW(
            "could not leave IPv4 multicast group '$2' via interface `$3`\n"
            "[`setsockopt()` failed: $1]",
            format_errno(), maddr, ifindex);
    }
    else if(maddr.family() == AF_INET6) {
      // IPv6
      ::ipv6_mreq mreq;
      mreq.ipv6mr_multiaddr = maddr.data().addr6.sin6_addr;
      mreq.ipv6mr_interface = static_cast<unsigned>(ifindex);
      if(::setsockopt(this->get_fd(), IPPROTO_IPV6, IPV6_DROP_MEMBERSHIP, &mreq, sizeof(mreq)) != 0)
        POSEIDON_THROW(
            "could not leave IPv6 multicast group '$2' via interface `$3`\n"
            "[`setsockopt()` failed: $1]",
            format_errno(), maddr, ifindex);
    }
    else
      POSEIDON_THROW("unsupported multicast address family `$1`", maddr.family());
  }

void
Abstract_Datagram_Socket::
leave_multicast_group(const Socket_Address& maddr, const char* ifname)
  {
    return this->leave_multicast_group(maddr, do_ifname_to_ifindex(ifname));
  }

}  // namespace poseidon
