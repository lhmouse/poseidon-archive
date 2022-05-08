// This file is part of Poseidon.
// Copyleft 2020, LH_Mouse. All wrongs reserved.

#include "../precompiled.ipp"
#include "abstract_udp_socket.hpp"
#include "../static/network_driver.hpp"
#include "../utils.hpp"
#include <net/if.h>

namespace poseidon {
namespace {

struct Queued_Packet_Header
  {
    uint16_t addrlen;
    uint16_t datalen;
  };

int
do_ifname_to_ifindex(const char* ifname)
  {
    int ifindex = 0;  // default
    if(ifname && *ifname) {
      // Note that valid indices start from one.
      ifindex = static_cast<int>(::if_nametoindex(ifname));
      if(ifindex == 0)
        POSEIDON_THROW("invalid network interface `$2`\n"
                       "[`$1()` failed: $1]",
                       format_errno(), ifname);
    }
    return ifindex;
  }

}  // namespace

Abstract_UDP_Socket::
Abstract_UDP_Socket(::sa_family_t family)
  : Abstract_Socket(family, SOCK_DGRAM, IPPROTO_UDP)
  {
  }

Abstract_UDP_Socket::
~Abstract_UDP_Socket()
  {
  }

IO_Result
Abstract_UDP_Socket::
do_socket_close_unlocked() noexcept
  {
    switch(this->m_cstate) {
      case connection_state_empty:
      case connection_state_connecting:
        // Shut down the connection. Discard pending data.
        ::shutdown(this->get_fd(), SHUT_RDWR);
        this->m_cstate = connection_state_closed;
        POSEIDON_LOG_TRACE("Marked UDP socket as CLOSED (not open): $1", this);
        return io_result_end_of_stream;

      case connection_state_established:
      case connection_state_closing: {
        // Ensure pending data are delivered.
        if(this->m_wqueue.size()) {
          this->m_cstate = connection_state_closing;
          POSEIDON_LOG_TRACE("Marked UDP socket as CLOSING (data pending): $1", this);
          return io_result_partial_work;
        }

        // Pending data have been cleared. Shut it down.
        ::shutdown(this->get_fd(), SHUT_RDWR);
        this->m_cstate = connection_state_closed;
        POSEIDON_LOG_TRACE("Marked UDP socket as CLOSED (pending data clear): $1", this);
        return io_result_end_of_stream;
      }

      case connection_state_closed:
        // Do nothing.
        return io_result_end_of_stream;

      default:
        ROCKET_ASSERT(false);
    }
  }

IO_Result
Abstract_UDP_Socket::
do_socket_on_poll_read(simple_mutex::unique_lock& lock, char* hint, size_t size)
  {
    lock.lock(this->m_io_mutex);
    if(this->m_cstate == connection_state_closed)
      return io_result_end_of_stream;

    try {
      // Try reading a packet.
      Socket_Address::storage addrst;
      ::socklen_t addrlen = sizeof(addrst);
      ::ssize_t nread = ::recvfrom(this->get_fd(), hint, size, 0, addrst, &addrlen);
      if(nread < 0)
        return get_io_result_from_errno("recvfrom", errno);

      // Process the packet that has been read.
      lock.unlock();
      Socket_Address addr(addrst, addrlen);
      this->do_socket_on_receive(addr, hint, static_cast<size_t>(nread));
    }
    catch(exception& stdex) {
      // It is probably bad to let the exception propagate to network driver and kill
      // this server socket... so we catch and ignore this exception.
      POSEIDON_LOG_ERROR("UDP socket read error: $1\n"
                         "[socket class `$2`]",
                         stdex, typeid(*this));
    }

    lock.lock(this->m_io_mutex);
    return io_result_partial_work;
  }

size_t
Abstract_UDP_Socket::
do_write_queue_size(simple_mutex::unique_lock& lock) const
  {
    lock.lock(this->m_io_mutex);

    // Get the size of pending data.
    // This is guaranteed to be zero when no data are to be sent.
    size_t size = this->m_wqueue.size();
    if(size != 0)
      return size;

    // If a shutdown request is pending, report at least one byte.
    return this->m_cstate == connection_state_closing;
  }

IO_Result
Abstract_UDP_Socket::
do_socket_on_poll_write(simple_mutex::unique_lock& lock, char* /*hint*/, size_t /*size*/)
  {
    lock.lock(this->m_io_mutex);
    if(this->m_cstate == connection_state_closed)
      return io_result_end_of_stream;

    // If the socket is in CONNECTING state, mark it ESTABLISHED.
    // Note UDP is connectless, so the socket is always writable unless some earlier error
    // has occurred, such as failure to bind it to a specific address.
    if(this->m_cstate < connection_state_established) {
      this->m_cstate = connection_state_established;

      lock.unlock();
      this->do_socket_on_establish();
    }
    lock.lock(this->m_io_mutex);

    try {
      // Try extracting a packet. This function shall match `do_socket_send()`.
      Queued_Packet_Header head;
      size_t size = this->m_wqueue.getn(reinterpret_cast<char*>(&head), sizeof(head));
      if((size == 0) && (this->m_cstate > connection_state_established))
        return this->do_socket_close_unlocked();

      if(size == 0)
        return io_result_end_of_stream;

      // Get the destination address.
      ROCKET_ASSERT(size == sizeof(head));
      Socket_Address::storage addrst;
      size = this->m_wqueue.getn(reinterpret_cast<char*>(&addrst), head.addrlen);
      ROCKET_ASSERT(size == head.addrlen);

      // Send the payload. No matter whether the operation succeeds or not, it will
      // always be removed from the send queue.
      ROCKET_ASSERT(this->m_wqueue.size() >= head.datalen);
      ::ssize_t nwritten = ::sendto(this->get_fd(),
                       this->m_wqueue.data(), head.datalen, 0, addrst, head.addrlen);
      this->m_wqueue.discard(head.datalen);
      if(nwritten < 0)
        return get_io_result_from_errno("sendto", errno);
    }
    catch(exception& stdex) {
      // It is probably bad to let the exception propagate to network driver and kill
      // this server socket... so we catch and ignore this exception.
      POSEIDON_LOG_ERROR("UDP socket write error: $1\n"
                         "[socket class `$2`]",
                         stdex, typeid(*this));
    }

    lock.lock(this->m_io_mutex);
    return io_result_partial_work;
  }

void
Abstract_UDP_Socket::
do_socket_on_poll_close(int err)
  {
    simple_mutex::unique_lock lock(this->m_io_mutex);
    this->m_cstate = connection_state_closed;
    lock.unlock();

    this->do_socket_on_close(err);
  }

void
Abstract_UDP_Socket::
do_socket_bind(const Socket_Address& addr)
  {
    static constexpr int yes[] = { -1 };
    int res = ::setsockopt(this->get_fd(), SOL_SOCKET, SO_REUSEADDR, yes, sizeof(yes));
    ROCKET_ASSERT(res == 0);

    if(::bind(this->get_fd(), addr.data(), addr.ssize()) != 0)
      POSEIDON_THROW("failed to bind UDP socket onto '$2'\n"
                     "[`bind()` failed: $1]",
                     format_errno(), addr);
  }

void
Abstract_UDP_Socket::
do_socket_on_establish()
  {
    POSEIDON_LOG_INFO("UDP socket listening: local '$1'",
                      this->get_local_address());
  }

void
Abstract_UDP_Socket::
do_socket_on_close(int err)
  {
    POSEIDON_LOG_INFO("UDP socket closed: local '$1', $2",
                      this->get_local_address(), format_errno(err));
  }

bool
Abstract_UDP_Socket::
do_socket_send(const Socket_Address& addr, const char* data, size_t size)
  {
    simple_mutex::unique_lock lock(this->m_io_mutex);
    if(this->m_cstate > connection_state_established)
      return false;

    // Reserve an exaggerate amount of space for the header, the destination
    // address and the largest payload.
    this->m_wqueue.reserve(0x11000);

    // Write the header.
    Queued_Packet_Header head;
    head.addrlen = static_cast<uint16_t>(addr.size());
    head.datalen = static_cast<uint16_t>(size);

    if(size != head.datalen)
      POSEIDON_LOG_WARN("UDP packet truncated (size `$1` too large)", size);

    ::std::memcpy(this->m_wqueue.mut_end(), &head, sizeof(head));
    this->m_wqueue.accept(sizeof(head));

    // Write the destination address.
    ::std::memcpy(this->m_wqueue.mut_end(), addr.data(), head.addrlen);
    this->m_wqueue.accept(head.addrlen);

    // Write the payload.
    ::std::memcpy(this->m_wqueue.mut_end(), data, head.datalen);
    this->m_wqueue.accept(head.datalen);
    lock.unlock();

    // Notify the driver about availability of outgoing data.
    Network_Driver::notify_writable_internal(*this);
    return true;
  }

void
Abstract_UDP_Socket::
set_multicast(int ifindex, uint8_t ttl, bool loop)
  {
    // Get the socket family.
    ::sa_family_t family;
    ::socklen_t addrlen = sizeof(family);
    if(::getsockname(this->get_fd(), reinterpret_cast<::sockaddr*>(&family), &addrlen) != 0)
      POSEIDON_THROW("failed to get socket family\n"
                     "[`getsockname()` failed: $1]",
                     format_errno());

    if(family == AF_INET) {
      // IPv4
      ::ip_mreqn mreq;
      mreq.imr_multiaddr.s_addr = INADDR_ANY;
      mreq.imr_address.s_addr = INADDR_ANY;
      mreq.imr_ifindex = ifindex;
      if(::setsockopt(this->get_fd(), IPPROTO_IP, IP_MULTICAST_IF,
                      &mreq, sizeof(mreq)) != 0)
        POSEIDON_THROW("failed to set IPv4 multicast interface to `$2`\n"
                       "[`setsockopt()` failed: $1]",
                       format_errno(), ifindex);

      int value = ttl;
      if(::setsockopt(this->get_fd(), IPPROTO_IP, IP_MULTICAST_TTL,
                      &value, sizeof(value)) != 0)
        POSEIDON_THROW("failed to set IPv4 multicast TTL to `$2`\n"
                       "[`setsockopt()` failed: $1]",
                       format_errno(), value);

      value = -loop;
      if(::setsockopt(this->get_fd(), IPPROTO_IP, IP_MULTICAST_LOOP,
                      &value, sizeof(value)) != 0)
        POSEIDON_THROW("failed to set IPv4 multicast loopback to `$2`\n"
                       "[`setsockopt()` failed: $1]",
                       format_errno(), value);
    }
    else if(family == AF_INET6) {
      // IPv6
      if(::setsockopt(this->get_fd(), IPPROTO_IPV6, IPV6_MULTICAST_IF,
                      &ifindex, sizeof(ifindex)) != 0)
        POSEIDON_THROW("failed to set IPv6 multicast interface to `$2`\n"
                       "[`setsockopt()` failed: $1]",
                       format_errno(), ifindex);

      int value = ttl;
      if(::setsockopt(this->get_fd(), IPPROTO_IPV6, IPV6_MULTICAST_HOPS,
                      &value, sizeof(value)) != 0)
        POSEIDON_THROW("failed to set IPv6 multicast TTL to `$2`\n"
                       "[`setsockopt()` failed: $1]",
                       format_errno(), value);

      value = -loop;
      if(::setsockopt(this->get_fd(), IPPROTO_IPV6, IPV6_MULTICAST_LOOP,
                      &value, sizeof(value)) != 0)
        POSEIDON_THROW("failed to set IPv6 multicast loopback to `$2`\n"
                       "[`setsockopt()` failed: $1]",
                       format_errno(), value);
    }
    else
      POSEIDON_THROW("unsupported address family `$1`", family);
  }

void
Abstract_UDP_Socket::
set_multicast(const char* ifname, uint8_t ttl, bool loop)
  {
    return this->set_multicast(do_ifname_to_ifindex(ifname), ttl, loop);
  }

void
Abstract_UDP_Socket::
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
      if(::setsockopt(this->get_fd(), IPPROTO_IP, IP_ADD_MEMBERSHIP,
                      &mreq, sizeof(mreq)) != 0)
        POSEIDON_THROW("failed to join IPv4 multicast group '$2' via interface `$3`\n"
                       "[`setsockopt()` failed: $1]",
                       format_errno(), maddr, ifindex);
    }
    else if(maddr.family() == AF_INET6) {
      // IPv6
      ::ipv6_mreq mreq;
      mreq.ipv6mr_multiaddr = maddr.data().addr6.sin6_addr;
      mreq.ipv6mr_interface = static_cast<unsigned>(ifindex);
      if(::setsockopt(this->get_fd(), IPPROTO_IPV6, IPV6_ADD_MEMBERSHIP,
                      &mreq, sizeof(mreq)) != 0)
        POSEIDON_THROW("failed to join IPv6 multicast group '$2' via interface `$3`\n"
                       "[`setsockopt()` failed: $1]",
                       format_errno(), maddr, ifindex);
    }
    else
      POSEIDON_THROW("unsupported multicast address family `$1`", maddr.family());
  }

void
Abstract_UDP_Socket::
join_multicast_group(const Socket_Address& maddr, const char* ifname)
  {
    return this->join_multicast_group(maddr, do_ifname_to_ifindex(ifname));
  }

void
Abstract_UDP_Socket::
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
      if(::setsockopt(this->get_fd(), IPPROTO_IP, IP_DROP_MEMBERSHIP,
                      &mreq, sizeof(mreq)) != 0)
        POSEIDON_THROW("failed to leave IPv4 multicast group '$2' via interface `$3`\n"
                       "[`setsockopt()` failed: $1]",
                       format_errno(), maddr, ifindex);
    }
    else if(maddr.family() == AF_INET6) {
      // IPv6
      ::ipv6_mreq mreq;
      mreq.ipv6mr_multiaddr = maddr.data().addr6.sin6_addr;
      mreq.ipv6mr_interface = static_cast<unsigned>(ifindex);
      if(::setsockopt(this->get_fd(), IPPROTO_IPV6, IPV6_DROP_MEMBERSHIP,
                      &mreq, sizeof(mreq)) != 0)
        POSEIDON_THROW("failed to leave IPv6 multicast group '$2' via interface `$3`\n"
                       "[`setsockopt()` failed: $1]",
                       format_errno(), maddr, ifindex);
    }
    else
      POSEIDON_THROW("unsupported multicast address family `$1`", maddr.family());
  }

void
Abstract_UDP_Socket::
leave_multicast_group(const Socket_Address& maddr, const char* ifname)
  {
    return this->leave_multicast_group(maddr, do_ifname_to_ifindex(ifname));
  }

bool
Abstract_UDP_Socket::
close() noexcept
  {
    simple_mutex::unique_lock lock(this->m_io_mutex);
    if(this->m_cstate > connection_state_established)
      return false;

    // Initiate asynchronous shutdown.
    this->do_socket_close_unlocked();
    lock.unlock();

    // Notify the driver about availability of outgoing data.
    Network_Driver::notify_writable_internal(*this);
    return true;
  }

}  // namespace poseidon
