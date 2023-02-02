// This file is part of Poseidon.
// Copyleft 2022, LH_Mouse. All wrongs reserved.

#include "../precompiled.ipp"
#include "tcp_socket.hpp"
#include "../static/async_logger.hpp"
#include "../utils.hpp"
#include <sys/socket.h>
#include <netinet/tcp.h>

namespace poseidon {

TCP_Socket::
TCP_Socket(unique_posix_fd&& fd)
  : Abstract_Socket(::std::move(fd))
  {
    // Use `TCP_NODELAY`. Errors are ignored.
    int ival = 1;
    ::setsockopt(this->fd(), IPPROTO_TCP, TCP_NODELAY, &ival, sizeof(ival));
  }

TCP_Socket::
TCP_Socket(const Socket_Address& saddr)
  : Abstract_Socket(SOCK_STREAM, IPPROTO_TCP)
  {
    // Use `TCP_NODELAY`. Errors are ignored.
    int ival = 1;
    ::setsockopt(this->fd(), IPPROTO_TCP, TCP_NODELAY, &ival, sizeof(ival));

    // Initiate a connection to `addr`.
    ::sockaddr_in6 addr;
    addr.sin6_family = AF_INET6;
    addr.sin6_port = htobe16(saddr.port());
    addr.sin6_flowinfo = 0;
    addr.sin6_addr = saddr.addr();
    addr.sin6_scope_id = 0;

    if((::connect(this->fd(), (const ::sockaddr*) &addr, sizeof(addr)) != 0) && (errno != EINPROGRESS))
      POSEIDON_THROW((
          "Failed to initiate TCP connection to `$4`",
          "[`connect()` failed: $3]",
          "[TCP socket `$1` (class `$2`)]"),
          this, typeid(*this), format_errno(), saddr);
  }

TCP_Socket::
~TCP_Socket()
  {
  }

void
TCP_Socket::
do_abstract_socket_on_closed(int err)
  {
    POSEIDON_LOG_INFO((
        "TCP connection to `$3` closed: $4",
        "[TCP socket `$1` (class `$2`)]"),
        this, typeid(*this), this->get_remote_address(), format_errno(err));
  }

void
TCP_Socket::
do_abstract_socket_on_readable()
  {
    recursive_mutex::unique_lock io_lock;
    auto& queue = this->do_abstract_socket_lock_read_queue(io_lock);
    size_t old_size = queue.size();
    ::ssize_t io_result = 0;

    for(;;) {
      // Read bytes and append them to `queue`.
      queue.reserve(0xFFFFU);
      io_result = ::recv(this->fd(), queue.mut_end(), queue.capacity(), 0);

      if(io_result < 0) {
        if((errno == EAGAIN) || (errno == EWOULDBLOCK))
          break;

        POSEIDON_LOG_ERROR((
            "Error reading TCP socket",
            "[`recv()` failed: $3]",
            "[TCP socket `$1` (class `$2`)]"),
            this, typeid(*this), format_errno());

        // The connection is now broken.
        this->quick_shut_down();
        return;
      }

      if(io_result == 0)
        break;

      // Accept incoming data.
      queue.accept((size_t) io_result);
    }

    if(old_size != queue.size())
      this->do_on_tcp_stream(queue);

    if(io_result == 0) {
      // If the end of stream has been reached, shut the connection down anyway.
      // Half-open connections are not supported.
      POSEIDON_LOG_INFO(("Closing TCP connection: remote = $1"), this->get_remote_address());
      ::shutdown(this->fd(), SHUT_RDWR);
    }
  }

void
TCP_Socket::
do_abstract_socket_on_writable()
  {
    recursive_mutex::unique_lock io_lock;
    auto& queue = this->do_abstract_socket_lock_write_queue(io_lock);
    ::ssize_t io_result = 0;

    for(;;) {
      // Write bytes from `queue` and remove those written.
      if(queue.size() == 0)
        break;

      io_result = ::send(this->fd(), queue.begin(), queue.size(), 0);

      if(io_result < 0) {
        if((errno == EAGAIN) || (errno == EWOULDBLOCK))
          break;

        POSEIDON_LOG_ERROR((
            "Error writing TCP socket",
            "[`send()` failed: $3]",
            "[TCP socket `$1` (class `$2`)]"),
            this, typeid(*this), format_errno());

        // The connection is now broken.
        this->quick_shut_down();
        return;
      }

      // Discard data that have been sent.
      queue.discard((size_t) io_result);
    }

    if(this->do_abstract_socket_set_state(socket_state_connecting, socket_state_established)) {
      // Deliver the establishment notification.
      POSEIDON_LOG_DEBUG(("TCP connection established: remote = $1"), this->get_remote_address());
      this->do_on_tcp_connected();
    }

    if(queue.empty() && this->do_abstract_socket_set_state(socket_state_closing, socket_state_closed)) {
      // If the socket has been marked closing and there are no more data, perform
      // complete shutdown.
      ::shutdown(this->fd(), SHUT_RDWR);
    }
  }

void
TCP_Socket::
do_on_tcp_connected()
  {
    POSEIDON_LOG_INFO((
        "TCP connection to `$3` established",
        "[TCP socket `$1` (class `$2`)]"),
        this, typeid(*this), this->get_remote_address());
  }

const cow_string&
TCP_Socket::
get_remote_address() const
  {
    if(this->m_peername_ready.load())
      return this->m_peername;

    // Try getting the address now.
    plain_mutex::unique_lock lock(this->m_peername_mutex);

    if(this->m_peername_ready.load())
      return this->m_peername;

    ::sockaddr_in6 addr;
    ::socklen_t addrlen = sizeof(addr);
    if(::getpeername(this->fd(), (::sockaddr*) &addr, &addrlen) != 0)
      return format(
          this->m_peername,  // reuse this buffer and return a copy
          "(invalid address: $1)", format_errno());

    if((addr.sin6_family != AF_INET6) || (addrlen != sizeof(addr)))
      return format(
          this->m_peername,  // reuse this buffer and return a copy
          "(address family unimplemented: $1)", addr.sin6_family);

    Socket_Address saddr;
    saddr.set_addr(addr.sin6_addr);
    saddr.set_port(be16toh(addr.sin6_port));

    this->m_peername = saddr.print_to_string();
    this->m_peername_ready.store(true);

    return this->m_peername;
  }

bool
TCP_Socket::
tcp_send(const char* data, size_t size)
  {
    if((data == nullptr) && (size != 0))
      POSEIDON_THROW((
          "Null data pointer",
          "[TCP socket `$1` (class `$2`)]"),
          this, typeid(*this));

    // If this socket has been marked closed, fail immediately.
    if(this->socket_state() >= socket_state_closing)
      return false;

    recursive_mutex::unique_lock io_lock;
    auto& queue = this->do_abstract_socket_lock_write_queue(io_lock);
    ::ssize_t io_result = 0;

    // Reserve backup space in case of partial writes.
    size_t nskip = 0;
    queue.reserve(size);

    if(queue.size() != 0) {
      // If there have been data pending, append new data to the end.
      ::memcpy(queue.mut_end(), data, size);
      queue.accept(size);
      return true;
    }

    for(;;) {
      // Try writing until the operation would block. This is essential for the
      // edge-triggered epoll to work reliably.
      if(nskip == size)
        break;

      io_result = ::send(this->fd(), data + nskip, size - nskip, 0);

      if(io_result < 0) {
        if((errno == EAGAIN) || (errno == EWOULDBLOCK))
          break;

        POSEIDON_LOG_ERROR((
            "Error writing TCP socket",
            "[`send()` failed: $3]",
            "[TCP socket `$1` (class `$2`)]"),
            this, typeid(*this), format_errno());

        // The connection is now broken.
        this->quick_shut_down();
        return false;
      }

      // Discard data that have been sent.
      nskip += (size_t) io_result;
    }

    // If the operation has completed only partially, buffer remaining data.
    // Space has already been reserved so this will not throw exceptions.
    ::memcpy(queue.mut_end(), data + nskip, size - nskip);
    queue.accept(size - nskip);
    return true;
  }

bool
TCP_Socket::
tcp_send(const linear_buffer& data)
  {
    return this->tcp_send(data.data(), data.size());
  }

bool
TCP_Socket::
tcp_send(const cow_string& data)
  {
    return this->tcp_send(data.data(), data.size());
  }

bool
TCP_Socket::
tcp_send(const string& data)
  {
    return this->tcp_send(data.data(), data.size());
  }

bool
TCP_Socket::
tcp_shut_down() noexcept
  {
    recursive_mutex::unique_lock io_lock;
    auto& queue = this->do_abstract_socket_lock_write_queue(io_lock);

    // If there are data pending, mark this socket as being closed. If a full
    // connection has been established, wait until all pending data to be sent.
    // The connection should be closed thereafter.
    if(!queue.empty() && this->do_abstract_socket_set_state(socket_state_established, socket_state_closing))
      return true;

    // If there are no data pending, shut it down immediately.
    return ::shutdown(this->fd(), SHUT_RDWR) == 0;
  }

}  // namespace poseidon
