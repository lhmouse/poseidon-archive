// This file is part of Poseidon.
// Copyleft 2022, LH_Mouse. All wrongs reserved.

#include "../precompiled.ipp"
#include "ssl_socket.hpp"
#include "../static/async_logger.hpp"
#include "../utils.hpp"
#include <sys/socket.h>
#include <netinet/tcp.h>
#include <openssl/ssl.h>
#include <openssl/err.h>

namespace poseidon {

SSL_Socket::
SSL_Socket(unique_posix_fd&& fd, const SSL_CTX_ptr& ssl_ctx)
  : Abstract_Socket(::std::move(fd)), m_ssl_ctx(ssl_ctx)
  {
    // Create the SSL structure.
    if(!this->m_ssl_ctx)
      POSEIDON_THROW((
          "Null SSL context pointer not valid",
          "[SSL socket `$1` (class `$2`)]"),
          this, typeid(*this));

    this->m_ssl.reset(::SSL_new(this->m_ssl_ctx));
    if(!this->m_ssl)
      POSEIDON_THROW((
          "Could not allocate server SSL structure",
          "[`SSL_new()` failed: $3]",
          "[SSL socket `$1` (class `$2`)]"),
          this, typeid(*this), ::ERR_reason_error_string(::ERR_peek_error()));

    if(!::SSL_set_fd(this->ssl(), this->fd()))
      POSEIDON_THROW((
          "Could not allocate SSL BIO for incoming connection",
          "[`SSL_set_fd()` failed: $3]",
          "[SSL socket `$1` (class `$2`)]"),
          this, typeid(*this), ::ERR_reason_error_string(::ERR_peek_error()));

    ::SSL_set_accept_state(this->ssl());

    // Use `TCP_NODELAY`. Errors are ignored.
    int ival = 1;
    ::setsockopt(this->fd(), IPPROTO_TCP, TCP_NODELAY, &ival, sizeof(ival));
  }

SSL_Socket::
SSL_Socket(int family, const SSL_CTX_ptr& ssl_ctx)
  : Abstract_Socket(family, SOCK_STREAM, IPPROTO_TCP), m_ssl_ctx(ssl_ctx)
  {
    // Create the SSL structure.
    if(!this->m_ssl_ctx)
      POSEIDON_THROW((
          "Null SSL context pointer not valid",
          "[SSL socket `$1` (class `$2`)]"),
          this, typeid(*this));

    this->m_ssl.reset(::SSL_new(this->m_ssl_ctx));
    if(!this->m_ssl)
      POSEIDON_THROW((
          "Could not allocate client SSL structure",
          "[`SSL_new()` failed: $3]",
          "[SSL socket `$1` (class `$2`)]"),
          this, typeid(*this), ::ERR_reason_error_string(::ERR_peek_error()));

    if(!::SSL_set_fd(this->ssl(), this->fd()))
      POSEIDON_THROW((
          "Could not allocate SSL BIO for outgoing connection",
          "[`SSL_set_fd()` failed: $3]",
          "[SSL socket `$1` (class `$2`)]"),
          this, typeid(*this), ::ERR_reason_error_string(::ERR_peek_error()));

    ::SSL_set_connect_state(this->ssl());

    // Use `TCP_NODELAY`. Errors are ignored.
    int ival = 1;
    ::setsockopt(this->fd(), IPPROTO_TCP, TCP_NODELAY, &ival, sizeof(ival));
  }

SSL_Socket::
~SSL_Socket()
  {
  }

void
SSL_Socket::
do_abstract_socket_on_readable()
  {
    recursive_mutex::unique_lock io_lock;
    auto& queue = this->do_abstract_socket_lock_read_queue(io_lock);

    // Try getting some bytes from this socket.
    queue.reserve(0xFFFFU);
    size_t datalen;

  try_io:
    datalen = queue.capacity();
    if(!::SSL_read_ex(this->ssl(), queue.mut_end(), datalen, &datalen)) {
      int ssl_err = ::SSL_get_error(this->ssl(), 0);
      if(is_any_of(ssl_err, { SSL_ERROR_WANT_READ, SSL_ERROR_WANT_WRITE }) && (errno == EINTR))
        goto try_io;

      // OpenSSL 1.1: EOF received without an SSL shutdown alert.
      if((ssl_err == SSL_ERROR_SYSCALL) && (errno == 0))
        ssl_err = SSL_ERROR_ZERO_RETURN;

      switch(ssl_err) {
        case SSL_ERROR_ZERO_RETURN:
          // Shut the connection down. Semi-open connections are not supported.
          // Send a close_notify alert, but don't wait for its response. If the
          // alert cannot be sent, ignore the error and force shutdown anyway.
          ssl_err = ::SSL_shutdown(this->ssl());
          POSEIDON_LOG_INFO(("Closing SSL connection: remote = $1, alert_received = $2"), this->get_remote_address(), ssl_err == 1);
          ::shutdown(this->fd(), SHUT_RDWR);
          return;

        case SSL_ERROR_WANT_READ:
        case SSL_ERROR_WANT_WRITE:
          return;

        case SSL_ERROR_SSL:
          // Shut the connection down due to an irrecoverable error, such as
          // when the peer requested SSLv3, or when the SSL connection was not
          // shut down properly.
          POSEIDON_LOG_INFO(("Closing SSL connection: remote = $1, reason = $2"), this->get_remote_address(), ::ERR_reason_error_string(::ERR_peek_error()));
          ::shutdown(this->fd(), SHUT_RDWR);
          return;

        case SSL_ERROR_SYSCALL:
          POSEIDON_THROW((
              "Error reading SSL socket",
              "[syscall failure: $3]",
              "[SSL socket `$1` (class `$2`)]"),
              this, typeid(*this), format_errno());
      }

      POSEIDON_THROW((
          "Error reading SSL socket",
          "[`SSL_read_ex()` failed: SSL error `$3`: $4]",
          "[SSL socket `$1` (class `$2`)]"),
          this, typeid(*this), ssl_err, ::ERR_reason_error_string(::ERR_peek_error()));
    }

    if(datalen == 0)
      return;

    // Accept these data.
    queue.accept(datalen);
    this->do_on_ssl_stream(queue);
  }

void
SSL_Socket::
do_abstract_socket_on_writable()
  {
    recursive_mutex::unique_lock io_lock;
    auto& queue = this->do_abstract_socket_lock_write_queue(io_lock);

    if(queue.empty())
      return;

    // Send some bytes from the write queue.
    size_t datalen;

  try_io:
    datalen = queue.size();
    if(!::SSL_write_ex(this->ssl(), queue.begin(), datalen, &datalen)) {
      int ssl_err = ::SSL_get_error(this->ssl(), 0);
      if(is_any_of(ssl_err, { SSL_ERROR_WANT_READ, SSL_ERROR_WANT_WRITE }) && (errno == EINTR))
        goto try_io;

      switch(ssl_err) {
        case SSL_ERROR_WANT_READ:
        case SSL_ERROR_WANT_WRITE:
          return;

        case SSL_ERROR_SSL:
          // Shut the connection down due to an irrecoverable error, such as
          // when the peer requested SSLv3, or when the SSL connection was not
          // shut down properly.
          POSEIDON_LOG_INFO(("Closing SSL connection: remote = $1, reason = $2"), this->get_remote_address(), ::ERR_reason_error_string(::ERR_peek_error()));
          ::shutdown(this->fd(), SHUT_RDWR);
          return;

        case SSL_ERROR_SYSCALL:
          POSEIDON_THROW((
              "Error writing SSL socket",
              "[syscall failure: $3]",
              "[SSL socket `$1` (class `$2`)]"),
              this, typeid(*this), format_errno());
      }

      POSEIDON_THROW((
          "Error writing SSL socket",
          "[`SSL_write_ex()` failed: SSL error `$3`: $4]",
          "[SSL socket `$1` (class `$2`)]"),
          this, typeid(*this), ssl_err, ::ERR_reason_error_string(::ERR_peek_error()));
    }

    // Remove sent bytes from the write queue.
    queue.discard(datalen);
  }

void
SSL_Socket::
do_abstract_socket_on_exception(exception& stdex)
  {
    this->quick_shut_down();

    POSEIDON_LOG_WARN((
        "SSL connection terminated due to exception: $3",
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
ssl_send(const linear_buffer& data)
  {
    return this->ssl_send(data.data(), data.size());
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

bool
SSL_Socket::
ssl_shut_down() noexcept
  {
    ::SSL_shutdown(this->ssl());
    return ::shutdown(this->fd(), SHUT_RDWR) == 0;
  }

}  // namespace poseidon
