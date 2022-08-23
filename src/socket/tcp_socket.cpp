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
  :
    Abstract_Socket(::std::move(fd))
  {
    // Use `TCP_NODELAY`. Errors are ignored.
    int ival = 1;
    ::setsockopt(this->fd(), IPPROTO_TCP, TCP_NODELAY, &ival, sizeof(ival));
  }

TCP_Socket::
TCP_Socket(const Socket_Address& addr)
  :
    Abstract_Socket(addr.family(), SOCK_STREAM, IPPROTO_TCP)
  {
    // Use `TCP_NODELAY`. Errors are ignored.
    int ival = 1;
    ::setsockopt(this->fd(), IPPROTO_TCP, TCP_NODELAY, &ival, sizeof(ival));

    if((::connect(this->fd(), addr.addr(), addr.ssize()) != 0) && (errno != EINPROGRESS))
      POSEIDON_THROW((
          "Failed to initiate TCP connection to `$4`",
          "[`connect()` failed: $3]",
          "[TCP socket `$1` (class `$2`)]"),
          this, typeid(*this), format_errno(), addr);
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
    ::ssize_t r;

    // Try getting some bytes from this socket.
  try_io:
    queue.reserve(0xFFFFU);
    r = ::recv(this->fd(), queue.mut_end(), queue.capacity(), 0);

    if(r == 0) {
      // Shut the connection down. Semi-open connections are not supported.
      POSEIDON_LOG_INFO(("Closing TCP connection: remote = $1"), this->get_remote_address());
      ::shutdown(this->fd(), SHUT_RDWR);
      return;
    }

    if(r < 0) {
      switch(errno) {
        case EINTR:
          goto try_io;

#if EWOULDBLOCK != EAGAIN
        case EAGAIN:
#endif
        case EWOULDBLOCK:
          return;
      }

      POSEIDON_THROW((
          "Error reading TCP socket",
          "[`recv()` failed: $3]",
          "[TCP socket `$1` (class `$2`)]"),
          this, typeid(*this), format_errno());
    }

    // Accept these data.
    queue.accept((size_t) r);
    this->do_on_tcp_stream(queue);
  }

void
TCP_Socket::
do_abstract_socket_on_writable()
  {
    recursive_mutex::unique_lock io_lock;
    auto& queue = this->do_abstract_socket_lock_write_queue(io_lock);
    ::ssize_t r;

    // Send some bytes from the write queue.
  try_io:
    r = 0;
    if(!queue.empty()) {
      r = ::send(this->fd(), queue.begin(), queue.size(), 0);
      if(r > 0)
        queue.discard((size_t) r);
    }

    if(r < 0) {
      switch(errno) {
        case EINTR:
          goto try_io;

#if EWOULDBLOCK != EAGAIN
        case EAGAIN:
#endif
        case EWOULDBLOCK:
          return;
      }

      POSEIDON_THROW((
          "Error writing TCP socket",
          "[`send()` failed: $3]",
          "[TCP socket `$1` (class `$2`)]"),
          this, typeid(*this), format_errno());
    }

    if(ROCKET_UNEXPECT(this->do_set_established())) {
      // Deliver the establishment notification.
      this->do_on_tcp_connected();
    }
  }

void
TCP_Socket::
do_abstract_socket_on_exception(exception& stdex)
  {
    this->quick_shut_down();

    POSEIDON_LOG_WARN((
        "TCP connection terminated due to exception: $3",
        "[TCP socket `$1` (class `$2`)]"),
        this, typeid(*this), stdex);
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
    if(!this->m_peername_ready.load()) {
      plain_mutex::unique_lock lock(this->m_peername_mutex);

      // Try getting the address.
      Socket_Address addr;
      ::socklen_t addrlen = (::socklen_t) addr.capacity();
      if(::getpeername(this->fd(), addr.mut_addr(), &addrlen) != 0)
        return this->m_peername = format_string("(error: $1)", format_errno());

      // Cache the result.
      addr.set_size(addrlen);
      this->m_peername = addr.format();
      this->m_peername_ready.store(true);
    }
    return this->m_peername;
  }

bool
TCP_Socket::
tcp_send(const char* data, size_t size)
  {
    if(size && !data)
      POSEIDON_THROW((
          "Null data pointer",
          "[TCP socket `$1` (class `$2`)]"),
          this, typeid(*this));

    // If this socket has been marked closed, fail immediately.
    if(this->socket_state() == socket_state_closed)
      return false;

    recursive_mutex::unique_lock io_lock;
    auto& queue = this->do_abstract_socket_lock_write_queue(io_lock);

    // Append data for sending.
    queue.putn(data, size);

    // Try writing once. This is essential for the edge-triggered epoll to work
    // reliably, because the level-triggered epoll does not check for `EPOLLOUT` by
    // default. If the packet has been sent anyway, discard it from the write queue.
    this->do_abstract_socket_on_writable();
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
    return ::shutdown(this->fd(), SHUT_RDWR) == 0;
  }

}  // namespace poseidon
