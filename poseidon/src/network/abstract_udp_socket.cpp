// This file is part of Poseidon.
// Copyleft 2020, LH_Mouse. All wrongs reserved.

#include "../precompiled.hpp"
#include "abstract_udp_socket.hpp"
#include "socket_address.hpp"
#include "../static/network_driver.hpp"
#include "../utilities.hpp"

namespace poseidon {
namespace {

IO_Result
do_translate_syscall_error(const char* func, int err)
  {
    if(err == EINTR)
      return io_result_not_eof;

    if(::rocket::is_any_of(err, { EAGAIN, EWOULDBLOCK }))
      return io_result_again;

    POSEIDON_THROW("UDP socket error\n"
                   "[`$1()` failed: $2]",
                   func, noadl::format_errno(err));
  }

struct Packet_Header
  {
    uint16_t addrlen;
    uint16_t datalen;
  };

void
do_wqueue_append(::rocket::linear_buffer& wqueue, const void* data, size_t size)
noexcept
  {
    ::std::memcpy(wqueue.mut_end(), data, size);
    wqueue.accept(size);
  }

}  // namespace

Abstract_UDP_Socket::
~Abstract_UDP_Socket()
  {
  }

IO_Result
Abstract_UDP_Socket::
do_async_shutdown_nolock()
noexcept
  {
    switch(this->m_cstate) {
      case connection_state_initial:
      case connection_state_connecting:
        // Shut down the connection. Discard pending data.
        ::shutdown(this->get_fd(), SHUT_RDWR);
        this->m_cstate = connection_state_closed;
        POSEIDON_LOG_TRACE("Marked UDP socket as CLOSED (not open): $1", this);
        return io_result_eof;

      case connection_state_established:
      case connection_state_closing: {
        // Ensure pending data are delivered.
        if(this->m_wqueue.size()) {
          this->m_cstate = connection_state_closing;
          POSEIDON_LOG_TRACE("Marked UDP socket as CLOSING (data pending): $1", this);
          return io_result_not_eof;
        }

        // Pending data have been cleared. Shut it down.
        ::shutdown(this->get_fd(), SHUT_RDWR);
        this->m_cstate = connection_state_closed;
        POSEIDON_LOG_TRACE("Marked UDP socket as CLOSED (pending data clear): $1", this);
        return io_result_eof;
      }

      case connection_state_closed:
        // Do nothing.
        return io_result_eof;

      default:
        ROCKET_ASSERT(false);
    }
  }

IO_Result
Abstract_UDP_Socket::
do_on_async_poll_read(Si_Mutex::unique_lock& lock, void* hint, size_t size)
  try {
    lock.assign(this->m_mutex);

    // If the socket is in CLOSED state, fail.
    if(this->m_cstate == connection_state_closed)
      return io_result_eof;

    // Try reading a packet.
    Socket_Address::storage_type addrst;
    Socket_Address::size_type addrlen = sizeof(addrst);
    ::ssize_t nread = ::recvfrom(this->get_fd(), hint, size, 0, addrst, &addrlen);
    if(nread < 0)
      return do_translate_syscall_error("recvfrom", errno);

    // Process the packet that has been read.
    lock.unlock();
    this->do_on_async_receive({ addrst, addrlen }, hint, static_cast<size_t>(nread));
    lock.assign(this->m_mutex);

    // Warning: Don't return `io_result_eof` i.e. zero.
    return io_result_not_eof;
  }
  catch(exception& stdex) {
    // It is probably bad to let the exception propagate to network driver and kill
    // this server socket... so we catch and ignore this exception.
    POSEIDON_LOG_ERROR("Error reading UDP socket: $2\n"
                       "[socket class `$1`]",
                       typeid(*this).name(), stdex.what());

    // Read other packets. The error is considered non-fatal.
    return io_result_not_eof;
  }

size_t
Abstract_UDP_Socket::
do_write_queue_size(Si_Mutex::unique_lock& lock)
const
  {
    lock.assign(this->m_mutex);

    // Get the size of pending data.
    // This is guaranteed to be zero when no data are to be sent.
    size_t size = this->m_wqueue.size();
    if(size != 0)
      return size;

    // If a shutdown request is pending, report at least one byte.
    if(this->m_cstate == connection_state_closing)
      return 1;

    // There is nothing to write.
    return 0;
  }

IO_Result
Abstract_UDP_Socket::
do_on_async_poll_write(Si_Mutex::unique_lock& lock, void* /*hint*/, size_t /*size*/)
  try {
    lock.assign(this->m_mutex);

    // If the socket is in CLOSED state, fail.
    if(this->m_cstate == connection_state_closed)
      return io_result_eof;

    // If the socket is in CONNECTING state, mark it ESTABLISHED.
    // Note UDP is connectless, so the socket is always writable unless some earlier error
    // has occurred, such as failure to bind it to a specific address.
    if(this->m_cstate < connection_state_established) {
      this->m_cstate = connection_state_established;

      lock.unlock();
      this->do_on_async_establish();
      lock.assign(this->m_mutex);
    }

    // Try extracting a packet.
    Packet_Header header;
    size_t size = this->m_wqueue.getn(reinterpret_cast<char (&)[]>(header), sizeof(header));
    if(size == 0) {
      if(this->m_cstate <= connection_state_established)
        return io_result_eof;

      // Shut down the connection completely now.
      return this->do_async_shutdown_nolock();
    }

    // Get the destination address.
    ROCKET_ASSERT(size == sizeof(header));
    ROCKET_ASSERT(this->m_wqueue.size() >= header.addrlen + header.datalen);

    Socket_Address::storage_type addrst;
    Socket_Address::size_type addrlen = header.addrlen;
    size = this->m_wqueue.getn(reinterpret_cast<char (&)[]>(addrst), addrlen);
    ROCKET_ASSERT(size == addrlen);

    // Write the payload, which shall be removed after `sendto()`, no matter whether it
    // succeeds or not.
    ::ssize_t nwritten = ::sendto(this->get_fd(), this->m_wqueue.data(), header.datalen, 0,
                                  addrst, addrlen);
    this->m_wqueue.discard(header.datalen);
    if(nwritten < 0)
      return do_translate_syscall_error("sendto", errno);

    // Warning: Don't return `io_result_eof` i.e. zero.
    return io_result_not_eof;
  }
  catch(exception& stdex) {
    // It is probably bad to let the exception propagate to network driver and kill
    // this server socket... so we catch and ignore this exception.
    POSEIDON_LOG_ERROR("Error writing UDP socket: $2\n"
                       "[socket class `$1`]",
                       typeid(*this).name(), stdex.what());

    // Write other packets. The error is considered non-fatal.
    return io_result_not_eof;
  }

void
Abstract_UDP_Socket::
do_on_async_poll_shutdown(int err)
  {
    Si_Mutex::unique_lock lock(this->m_mutex);
    this->m_cstate = connection_state_closed;
    lock.unlock();

    this->do_on_async_shutdown(err);
  }

void
Abstract_UDP_Socket::
do_on_async_establish()
  {
    POSEIDON_LOG_INFO("UDP socket opened: local '$1'",
                      this->get_local_address());
  }

void
Abstract_UDP_Socket::
do_on_async_shutdown(int err)
  {
    POSEIDON_LOG_INFO("UDP socket closed: local '$1', $2",
                      this->get_local_address(), noadl::format_errno(err));
  }

void
Abstract_UDP_Socket::
bind(const Socket_Address& addr)
  {
    // Bind onto `addr`.
    if(::bind(this->get_fd(), addr.data(), addr.size()) != 0)
      POSEIDON_THROW("failed to bind UDP socket onto '$2'\n"
                     "[`bind()` failed: $1]",
                     noadl::format_errno(errno), addr);

    POSEIDON_LOG_INFO("UDP socket listening on '$1'", this->get_local_address());
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
                     noadl::format_errno(errno));

    if(family == AF_INET) {
      // IPv4
      ::ip_mreqn mreq;
      mreq.imr_multiaddr.s_addr = INADDR_ANY;
      mreq.imr_address.s_addr = INADDR_ANY;
      mreq.imr_ifindex = ifindex;
      if(::setsockopt(this->get_fd(), IPPROTO_IP, IP_MULTICAST_IF, &mreq, sizeof(mreq)) != 0)
        POSEIDON_THROW("failed to set IPv4 multicast interface to `$2`\n"
                       "[`setsockopt()` failed: $1]",
                       noadl::format_errno(errno), ifindex);

      int value = ttl;
      if(::setsockopt(this->get_fd(), IPPROTO_IP, IP_MULTICAST_TTL, &value, sizeof(value)) != 0)
        POSEIDON_THROW("failed to set IPv4 multicast TTL to `$2`\n"
                       "[`setsockopt()` failed: $1]",
                       noadl::format_errno(errno), value);

      value = -loop;
      if(::setsockopt(this->get_fd(), IPPROTO_IP, IP_MULTICAST_LOOP, &value, sizeof(value)) != 0)
        POSEIDON_THROW("failed to set IPv4 multicast loopback to `$2`\n"
                       "[`setsockopt()` failed: $1]",
                       noadl::format_errno(errno), value);
    }
    else if(family == AF_INET6) {
      // IPv6
      if(::setsockopt(this->get_fd(), IPPROTO_IPV6, IPV6_MULTICAST_IF, &ifindex, sizeof(ifindex)) != 0)
        POSEIDON_THROW("failed to set IPv6 multicast interface to `$2`\n"
                       "[`setsockopt()` failed: $1]",
                       noadl::format_errno(errno), ifindex);

      int value = ttl;
      if(::setsockopt(this->get_fd(), IPPROTO_IPV6, IPV6_MULTICAST_HOPS, &value, sizeof(value)) != 0)
        POSEIDON_THROW("failed to set IPv6 multicast TTL to `$2`\n"
                       "[`setsockopt()` failed: $1]",
                       noadl::format_errno(errno), value);

      value = -loop;
      if(::setsockopt(this->get_fd(), IPPROTO_IPV6, IPV6_MULTICAST_LOOP, &value, sizeof(value)) != 0)
        POSEIDON_THROW("failed to set IPv6 multicast loopback to `$2`\n"
                       "[`setsockopt()` failed: $1]",
                       noadl::format_errno(errno), value);
    }
    else
      POSEIDON_THROW("unsupported address family `$1`", family);
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
      if(::setsockopt(this->get_fd(), IPPROTO_IP, IP_ADD_MEMBERSHIP, &mreq, sizeof(mreq)) != 0)
        POSEIDON_THROW("failed to join IPv4 multicast group '$2' via interface `$3`\n"
                       "[`setsockopt()` failed: $1]",
                       noadl::format_errno(errno), maddr, ifindex);
    }
    else if(maddr.family() == AF_INET6) {
      // IPv6
      ::ipv6_mreq mreq;
      mreq.ipv6mr_multiaddr = maddr.data().addr6.sin6_addr;
      mreq.ipv6mr_interface = static_cast<unsigned>(ifindex);
      if(::setsockopt(this->get_fd(), IPPROTO_IPV6, IPV6_ADD_MEMBERSHIP, &mreq, sizeof(mreq)) != 0)
        POSEIDON_THROW("failed to join IPv6 multicast group '$2' via interface `$3`\n"
                       "[`setsockopt()` failed: $1]",
                       noadl::format_errno(errno), maddr, ifindex);
    }
    else
      POSEIDON_THROW("unsupported multicast address family `$1`", maddr.family());
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
      if(::setsockopt(this->get_fd(), IPPROTO_IP, IP_DROP_MEMBERSHIP, &mreq, sizeof(mreq)) != 0)
        POSEIDON_THROW("failed to leave IPv4 multicast group '$2' via interface `$3`\n"
                       "[`setsockopt()` failed: $1]",
                       noadl::format_errno(errno), maddr, ifindex);
    }
    else if(maddr.family() == AF_INET6) {
      // IPv6
      ::ipv6_mreq mreq;
      mreq.ipv6mr_multiaddr = maddr.data().addr6.sin6_addr;
      mreq.ipv6mr_interface = static_cast<unsigned>(ifindex);
      if(::setsockopt(this->get_fd(), IPPROTO_IPV6, IPV6_DROP_MEMBERSHIP, &mreq, sizeof(mreq)) != 0)
        POSEIDON_THROW("failed to leave IPv6 multicast group '$2' via interface `$3`\n"
                       "[`setsockopt()` failed: $1]",
                       noadl::format_errno(errno), maddr, ifindex);
    }
    else
      POSEIDON_THROW("unsupported multicast address family `$1`", maddr.family());
  }

bool
Abstract_UDP_Socket::
async_send(const Socket_Address& addr, const void* data, size_t size)
  {
    Si_Mutex::unique_lock lock(this->m_mutex);
    if(this->m_cstate > connection_state_established)
      return false;

    // Append data to the write queue.
    Packet_Header header;
    header.addrlen = static_cast<uint16_t>(addr.size());
    header.datalen = static_cast<uint16_t>(::std::min<size_t>(size, UINT16_MAX));
    if(size != header.datalen)
      POSEIDON_LOG_WARN("UDP packet truncated (size `$1` too large)", size);

    this->m_wqueue.reserve(sizeof(header) + header.addrlen + header.datalen);
    do_wqueue_append(this->m_wqueue, &header, sizeof(header));
    do_wqueue_append(this->m_wqueue, addr.data(), header.addrlen);
    do_wqueue_append(this->m_wqueue, data, header.datalen);
    lock.unlock();

    // Notify the driver about availability of outgoing data.
    Network_Driver::notify_writable_internal(this);
    return true;
  }

bool
Abstract_UDP_Socket::
async_shutdown()
noexcept
  {
    Si_Mutex::unique_lock lock(this->m_mutex);
    if(this->m_cstate > connection_state_established)
      return false;

    // Initiate asynchronous shutdown.
    this->do_async_shutdown_nolock();
    lock.unlock();

    // Notify the driver about availability of outgoing data.
    Network_Driver::notify_writable_internal(this);
    return true;
  }

}  // namespace poseidon
