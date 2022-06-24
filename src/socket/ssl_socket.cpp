// This file is part of Poseidon.
// Copyleft 2022, LH_Mouse. All wrongs reserved.

#include "../precompiled.ipp"
#include "ssl_socket.hpp"
#include "../static/async_logger.hpp"
#include "../utils.hpp"
#include <sys/socket.h>

namespace poseidon {

SSL_Socket::
SSL_Socket(unique_posix_fd&& fd)
  : Abstract_Socket(::std::move(fd))
  {
  }

SSL_Socket::
SSL_Socket(int family)
  : Abstract_Socket(family, SOCK_STREAM, IPPROTO_TCP)
  {
  }

SSL_Socket::
~SSL_Socket()
  {
  }

IO_Result
SSL_Socket::
do_abstract_socket_on_readable()
  {
    recursive_mutex::unique_lock io_lock;
    auto& queue = this->do_abstract_socket_lock_read_queue(io_lock);

    // Try getting some bytes from this socket.
    queue.reserve(0xFFFFU);
    size_t datalen = queue.capacity();

    ::ssize_t r;
  try_io:
    r = ::recv(this->fd(), queue.mut_end(), datalen, 0);
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
          "Error reading SSL socket",
          "[`recv()` failed: $3]",
          "[SSL socket `$1` (class `$2`)]"),
          this, typeid(*this), format_errno());
    }

    if(r == 0)
      return io_result_end_of_file;

    // Accept these data.
    datalen = (size_t) r;
    queue.accept(datalen);

    this->do_on_ssl_stream(queue);
    return io_result_partial;
  }

IO_Result
SSL_Socket::
do_abstract_socket_on_writable()
  {
    recursive_mutex::unique_lock io_lock;
    auto& queue = this->do_abstract_socket_lock_write_queue(io_lock);

    if(queue.empty())
      return io_result_partial;

    // Send some bytes from the write queue.
    size_t datalen = queue.size();
    ::ssize_t r;
  try_io:
    r = ::send(this->fd(), queue.begin(), datalen, 0);
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
          "Error writing SSL socket",
          "[`send()` failed: $3]",
          "[SSL socket `$1` (class `$2`)]"),
          this, typeid(*this), format_errno());
    }

    // Remove sent bytes from the write queue.
    queue.discard(datalen);
    return io_result_partial;
  }

void
SSL_Socket::
do_abstract_socket_on_exception(exception& stdex)
  {
    this->abort();

    POSEIDON_LOG_WARN((
        "Aborting connection due to exception: $3",
        "[SSL socket `$1` (class `$2`)]"),
        this, typeid(*this), stdex);
  }

const Socket_Address&
SSL_Socket::
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
SSL_Socket::
ssl_send(const char* data, size_t size)
  {
    if(size && !data)
      POSEIDON_THROW((
          "Null data pointer",
          "[SSL socket `$1` (class `$2`)]"),
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
SSL_Socket::
ssl_send(const cow_string& data)
  {
    return this->ssl_send(data.data(), data.size());
  }

bool
SSL_Socket::
ssl_send(const string& data)
  {
    return this->ssl_send(data.data(), data.size());
  }

}  // namespace poseidon
