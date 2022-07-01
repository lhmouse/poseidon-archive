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
TCP_Socket(int family)
  : Abstract_Socket(family, SOCK_STREAM, IPPROTO_TCP)
  {
    // Use `TCP_NODELAY`. Errors are ignored.
    int ival = 1;
    ::setsockopt(this->fd(), IPPROTO_TCP, TCP_NODELAY, &ival, sizeof(ival));
  }

TCP_Socket::
~TCP_Socket()
  {
  }

IO_Result
TCP_Socket::
do_abstract_socket_on_readable()
  {
    recursive_mutex::unique_lock io_lock;
    auto& queue = this->do_abstract_socket_lock_read_queue(io_lock);

    // Try getting some bytes from this socket.
    queue.reserve(0xFFFFU);
    size_t datalen;

  try_io:
    datalen = queue.capacity();
    ::ssize_t r = ::recv(this->fd(), queue.mut_end(), datalen, 0);
    datalen = (size_t) r;

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
          "Error reading TCP socket",
          "[`recv()` failed: $3]",
          "[TCP socket `$1` (class `$2`)]"),
          this, typeid(*this), format_errno());
    }

    if(r == 0)
      return io_result_end_of_file;

    // Accept these data.
    queue.accept(datalen);

    this->do_on_tcp_stream(queue);
    return io_result_partial;
  }

IO_Result
TCP_Socket::
do_abstract_socket_on_writable()
  {
    recursive_mutex::unique_lock io_lock;
    auto& queue = this->do_abstract_socket_lock_write_queue(io_lock);

    if(queue.empty())
      return io_result_partial;

    // Send some bytes from the write queue.
    size_t datalen;

  try_io:
    datalen = queue.size();
    ::ssize_t r = ::send(this->fd(), queue.begin(), datalen, 0);
    datalen = (size_t) r;

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
          "Error writing TCP socket",
          "[`send()` failed: $3]",
          "[TCP socket `$1` (class `$2`)]"),
          this, typeid(*this), format_errno());
    }

    // Remove sent bytes from the write queue.
    queue.discard(datalen);
    return io_result_partial;
  }

void
TCP_Socket::
do_abstract_socket_on_exception(exception& stdex)
  {
    this->abort();

    POSEIDON_LOG_WARN((
        "Aborting connection due to exception: $3",
        "[TCP socket `$1` (class `$2`)]"),
        this, typeid(*this), stdex);
  }

const Socket_Address&
TCP_Socket::
get_remote_address() const
  {
    // Get the socket name and cache it.
    this->m_peername_once.call(
      [this] {
        ::socklen_t addrlen = (::socklen_t) this->m_peername.capacity();
        if(::getpeername(this->fd(), this->m_peername.mut_addr(), &addrlen) != 0)
          POSEIDON_THROW((
              "Could not get local address of socket",
              "[`getpeername()` failed: $1]"),
              format_errno());

        // Accept the address.
        this->m_peername.set_size(addrlen);
      });
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

}  // namespace poseidon
