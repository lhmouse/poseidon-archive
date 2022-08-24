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
    // Create a new non-blocking socket.
    Abstract_Socket(::std::move(fd))
  {
    // Use `TCP_NODELAY`. Errors are ignored.
    int ival = 1;
    ::setsockopt(this->fd(), IPPROTO_TCP, TCP_NODELAY, &ival, sizeof(ival));
  }

TCP_Socket::
TCP_Socket(const Socket_Address& addr)
  :
    // Take ownership of an existent socket.
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
    plain_mutex::unique_lock io_lock;
    auto& queue = this->do_abstract_socket_lock_read_queue(io_lock);
    ::ssize_t io_result = 0;

    for(;;) {
      // Read bytes and append them to `queue`.
      queue.reserve(0xFFFFU);
      io_result = ::recv(this->fd(), queue.mut_end(), queue.capacity(), 0);

      if(io_result < 0) {
        if(errno == EINTR)
          continue;

        if((errno == EAGAIN) || (errno == EWOULDBLOCK))
          break;

        POSEIDON_THROW((
            "Error reading TCP socket",
            "[`recv()` failed: $3]",
            "[TCP socket `$1` (class `$2`)]"),
            this, typeid(*this), format_errno());
      }

      if(io_result == 0)
        break;

      // Accept incoming data.
      queue.accept((size_t) io_result);
    }

    this->do_on_tcp_stream(queue);

    if(io_result != 0)
      return;

    // If the end of stream has been reached, shut the connection down anyway.
    // Half-open connections are not supported.
    POSEIDON_LOG_INFO(("Closing TCP connection: remote = $1"), this->get_remote_address());
    ::shutdown(this->fd(), SHUT_RDWR);
  }

void
TCP_Socket::
do_abstract_socket_on_writable()
  {
    plain_mutex::unique_lock io_lock;
    auto& queue = this->do_abstract_socket_lock_write_queue(io_lock);
    ::ssize_t io_result = 0;

    for(;;) {
      // Write bytes from `queue` and remove those written.
      if(queue.size() == 0)
        break;

      io_result = ::send(this->fd(), queue.begin(), queue.size(), 0);

      if(io_result < 0) {
        if(errno == EINTR)
          continue;

        if((errno == EAGAIN) || (errno == EWOULDBLOCK) || (errno == EPIPE))
          break;

        POSEIDON_THROW((
            "Error writing TCP socket",
            "[`send()` failed: $3]",
            "[TCP socket `$1` (class `$2`)]"),
            this, typeid(*this), format_errno());
      }

      // Discard data that have been sent.
      queue.discard((size_t) io_result);
    }

    if(!this->do_set_established())
      return;

    // Deliver the establishment notification.
    POSEIDON_LOG_DEBUG(("Established TCP connection: remote = $1"), this->get_remote_address());
    this->do_on_tcp_connected();
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
        return format(this->m_peername, "(unknown address: $1)", format_errno());

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
    if((data == nullptr) && (size != 0))
      POSEIDON_THROW((
          "Null data pointer",
          "[TCP socket `$1` (class `$2`)]"),
          this, typeid(*this));

    // If this socket has been marked closed, fail immediately.
    if(this->socket_state() == socket_state_closed)
      return false;

    plain_mutex::unique_lock io_lock;
    auto& queue = this->do_abstract_socket_lock_write_queue(io_lock);
    ::ssize_t io_result = 0;

    // Append data to the write queue.
    // If there were no pending data, try writing once. This is essential for
    // the edge-triggered epoll to work reliably.
    queue.putn(data, size);
    if(ROCKET_EXPECT(queue.size() != size))
      return true;

    for(;;) {
      // Write bytes from `queue` and remove those written.
      if(queue.size() == 0)
        break;

      io_result = ::send(this->fd(), queue.begin(), queue.size(), 0);

      if(io_result < 0) {
        if(errno == EINTR)
          continue;

        if((errno == EAGAIN) || (errno == EWOULDBLOCK) || (errno == EPIPE))
          break;

        POSEIDON_THROW((
            "Error writing TCP socket",
            "[`send()` failed: $3]",
            "[TCP socket `$1` (class `$2`)]"),
            this, typeid(*this), format_errno());
      }

      // Discard data that have been sent.
      queue.discard((size_t) io_result);
    }
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
