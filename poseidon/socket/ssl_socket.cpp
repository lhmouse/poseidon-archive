// This file is part of Poseidon.
// Copyleft 2022, LH_Mouse. All wrongs reserved.

#include "../precompiled.ipp"
#include "ssl_socket.hpp"
#include "ssl_ctx_ptr.hpp"
#include "../static/async_logger.hpp"
#include "../utils.hpp"
#include <sys/socket.h>
#include <netinet/tcp.h>
#include <openssl/ssl.h>
#include <openssl/err.h>

namespace poseidon {

SSL_Socket::
SSL_Socket(unique_posix_fd&& fd, const SSL_CTX_ptr& ssl_ctx)
  : Abstract_Socket(::std::move(fd))
  {
    // Create an SSL structure in server mode.
    if(!ssl_ctx)
      POSEIDON_THROW((
          "Null SSL context pointer not valid",
          "[SSL socket `$1` (class `$2`)]"),
          this, typeid(*this));

    if(!this->m_ssl.reset(::SSL_new(ssl_ctx)))
      POSEIDON_THROW((
          "Could not allocate server SSL structure",
          "[`SSL_new()` failed: $3]",
          "[SSL socket `$1` (class `$2`)]"),
          this, typeid(*this), ::ERR_reason_error_string(::ERR_peek_error()));

    ::SSL_set_accept_state(this->ssl());

    if(!::SSL_set_fd(this->ssl(), this->fd()))
      POSEIDON_THROW((
          "Could not allocate SSL BIO for incoming connection",
          "[`SSL_set_fd()` failed: $3]",
          "[SSL socket `$1` (class `$2`)]"),
          this, typeid(*this), ::ERR_reason_error_string(::ERR_peek_error()));

    // Use `TCP_NODELAY`. Errors are ignored.
    int ival = 1;
    ::setsockopt(this->fd(), IPPROTO_TCP, TCP_NODELAY, &ival, sizeof(ival));
  }

SSL_Socket::
SSL_Socket(const Socket_Address& saddr, const SSL_CTX_ptr& ssl_ctx)
  : Abstract_Socket(SOCK_STREAM, IPPROTO_TCP)
  {
    // Create an SSL structure in client mode.
    if(!ssl_ctx)
      POSEIDON_THROW((
          "Null SSL context pointer not valid",
          "[SSL socket `$1` (class `$2`)]"),
          this, typeid(*this));

    if(!this->m_ssl.reset(::SSL_new(ssl_ctx)))
      POSEIDON_THROW((
          "Could not allocate client SSL structure",
          "[`SSL_new()` failed: $3]",
          "[SSL socket `$1` (class `$2`)]"),
          this, typeid(*this), ::ERR_reason_error_string(::ERR_peek_error()));

    ::SSL_set_connect_state(this->ssl());

    if(!::SSL_set_fd(this->ssl(), this->fd()))
      POSEIDON_THROW((
          "Could not allocate SSL BIO for outgoing connection",
          "[`SSL_set_fd()` failed: $3]",
          "[SSL socket `$1` (class `$2`)]"),
          this, typeid(*this), ::ERR_reason_error_string(::ERR_peek_error()));

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
          "Failed to initiate SSL connection to `$4`",
          "[`connect()` failed: $3]",
          "[SSL socket `$1` (class `$2`)]"),
          this, typeid(*this), format_errno(), saddr);
  }

SSL_Socket::
~SSL_Socket()
  {
  }

charbuf_256
SSL_Socket::
do_on_ssl_alpn_request(cow_vector<charbuf_256>&& protos)
  {
    (void) protos;
    return { };
  }

void
SSL_Socket::
do_ssl_alpn_request(const charbuf_256* protos_opt, size_t protos_size)
  {
    // Generate the list of protocols in wire format.
    linear_buffer pbuf;

    for(size_t k = 0;  k != protos_size;  ++k) {
      const char* str = protos_opt[k];

      // Empty protocol names are ignored.
      size_t len = ::strlen(str);
      if(len == 0)
        continue;

      ROCKET_ASSERT(len <= 255);
      pbuf.putc((char) len);
      pbuf.putn(str, len);
      POSEIDON_LOG_TRACE(("Requesting ALPN protocol: $1"), str);
    }

    if(::SSL_set_alpn_protos(this->ssl(), (const uint8_t*) pbuf.data(), (uint32_t) pbuf.size()) != 0)
      POSEIDON_THROW((
          "Failed to set ALPN protocol list",
          "[`SSL_set_alpn_protos()` failed]",
          "[SSL socket `$1` (class `$2`)]"),
          this, typeid(*this));
  }

void
SSL_Socket::
do_ssl_alpn_request(const cow_vector<charbuf_256>& protos)
  {
    this->do_ssl_alpn_request(protos.data(), protos.size());
  }

void
SSL_Socket::
do_ssl_alpn_request(initializer_list<charbuf_256> protos)
  {
    this->do_ssl_alpn_request(protos.begin(), protos.size());
  }

void
SSL_Socket::
do_ssl_alpn_request(const charbuf_256& proto)
  {
    this->do_ssl_alpn_request(&proto, 1);
  }

void
SSL_Socket::
do_abstract_socket_on_closed(int err)
  {
    POSEIDON_LOG_INFO((
        "SSL connection to `$3` closed: $4",
        "[SSL socket `$1` (class `$2`)]"),
        this, typeid(*this), this->remote_address(), format_errno(err));
  }

void
SSL_Socket::
do_abstract_socket_on_readable()
  {
    recursive_mutex::unique_lock io_lock;
    auto& queue = this->do_abstract_socket_lock_read_queue(io_lock);
    size_t old_size = queue.size();
    int ssl_err = 0;

    for(;;) {
      // Read bytes and append them to `queue`.
      queue.reserve(0xFFFFU);
      ssl_err = 0;
      size_t datalen;
      int ret = ::SSL_read_ex(this->ssl(), queue.mut_end(), queue.capacity(), &datalen);

      if(ret == 0) {
        ssl_err = ::SSL_get_error(this->ssl(), ret);

        // Check for EOF without a shutdown alert.
        if((ssl_err == SSL_ERROR_SYSCALL) && (errno == 0))
          ssl_err = SSL_ERROR_ZERO_RETURN;

#ifdef SSL_R_UNEXPECTED_EOF_WHILE_READING
        if((ssl_err == SSL_ERROR_SSL) && (ERR_GET_REASON(::ERR_peek_error()) == SSL_R_UNEXPECTED_EOF_WHILE_READING))
          ssl_err = SSL_ERROR_ZERO_RETURN;
#endif  // Open SSL 3.0

        if((ssl_err == SSL_ERROR_ZERO_RETURN) || (ssl_err == SSL_ERROR_WANT_READ) || (ssl_err == SSL_ERROR_WANT_WRITE))
          break;

        POSEIDON_LOG_ERROR((
            "Error reading SSL socket",
            "[`SSL_read_ex()` failed: SSL_get_error = `$3`, ERR_peek_error = `$4`, errno = `$5`]",
            "[SSL socket `$1` (class `$2`)]"),
            this, typeid(*this), ssl_err, ::ERR_reason_error_string(::ERR_peek_error()), format_errno());

        // The connection is now broken.
        this->quick_shut_down();
        return;
      }

      // Accept incoming data.
      queue.accept(datalen);
    }

    const char* alpn_str;
    unsigned int alpn_len;
    ::SSL_get0_alpn_selected(this->ssl(), (const uint8_t**) &alpn_str, &alpn_len);

    if(alpn_len != 0)
      this->m_alpn_proto.assign(alpn_str, alpn_len);

    if(old_size != queue.size())
      this->do_on_ssl_stream(queue);

    if(ssl_err == SSL_ERROR_ZERO_RETURN) {
      // If the end of stream has been reached, shut the connection down anyway.
      // Half-open connections are not supported.
      bool alerted = ::SSL_shutdown(this->ssl()) == 1;
      POSEIDON_LOG_INFO(("Closing SSL connection: remote = $1, alerted = $2"), this->remote_address(), alerted);
      ::shutdown(this->fd(), SHUT_RDWR);
    }
  }

void
SSL_Socket::
do_abstract_socket_on_oob_readable()
  {
    char data;
    ::ssize_t io_result;

    // If there are no OOB data, `recv()` fails with `EINVAL`.
    io_result = ::recv(this->fd(), &data, 1, MSG_OOB);
    if(io_result <= 0)
      return;

    // Accept it.
    this->do_on_ssl_oob_byte(data);
  }

void
SSL_Socket::
do_abstract_socket_on_writable()
  {
    recursive_mutex::unique_lock io_lock;
    auto& queue = this->do_abstract_socket_lock_write_queue(io_lock);
    int ssl_err = 0;

    for(;;) {
      // Write bytes from `queue` and remove those written.
      if(queue.size() == 0)
        break;

      ssl_err = 0;
      size_t datalen;
      int ret = ::SSL_write_ex(this->ssl(), queue.begin(), queue.size(), &datalen);

      if(ret == 0) {
        ssl_err = ::SSL_get_error(this->ssl(), ret);

        if((ssl_err == SSL_ERROR_WANT_READ) || (ssl_err == SSL_ERROR_WANT_WRITE))
          break;

        POSEIDON_LOG_ERROR((
            "Error writing SSL socket",
            "[`SSL_write_ex()` failed: SSL_get_error = `$3`, ERR_peek_error = `$4`, errno = `$5`]",
            "[SSL socket `$1` (class `$2`)]"),
            this, typeid(*this), ssl_err, ::ERR_reason_error_string(::ERR_peek_error()), format_errno());

        // The connection is now broken.
        this->quick_shut_down();
        return;
      }

      // Discard data that have been sent.
      queue.discard(datalen);
    }

    if(this->do_abstract_socket_set_state(socket_state_connecting, socket_state_established)) {
      // Deliver the establishment notification.
      POSEIDON_LOG_DEBUG(("SSL connection established: remote = $1"), this->remote_address());
      this->do_on_ssl_connected();
    }

    if(queue.empty() && this->do_abstract_socket_set_state(socket_state_closing, socket_state_closed)) {
      // If the socket has been marked closing and there are no more data, perform
      // complete shutdown.
      ::SSL_shutdown(this->ssl());
      ::shutdown(this->fd(), SHUT_RDWR);
    }
  }

void
SSL_Socket::
do_on_ssl_connected()
  {
    POSEIDON_LOG_INFO((
        "SSL connection to `$3` established",
        "[SSL socket `$1` (class `$2`)]"),
        this, typeid(*this), this->remote_address());
  }

void
SSL_Socket::
do_on_ssl_oob_byte(char data)
  {
    POSEIDON_LOG_INFO((
        "SSL connection received out-of-band data: $3 ($4)",
        "[SSL socket `$1` (class `$2`)]"),
        this, typeid(*this), (int) data, (char) data);
  }

const cow_string&
SSL_Socket::
remote_address() const
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
SSL_Socket::
ssl_send(const char* data, size_t size)
  {
    if((data == nullptr) && (size != 0))
      POSEIDON_THROW((
          "Null data pointer",
          "[SSL socket `$1` (class `$2`)]"),
          this, typeid(*this));

    // If this socket has been marked closed, fail immediately.
    if(this->socket_state() >= socket_state_closing)
      return false;

    recursive_mutex::unique_lock io_lock;
    auto& queue = this->do_abstract_socket_lock_write_queue(io_lock);
    int ssl_err = 0;

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

      ssl_err = 0;
      size_t datalen;
      int ret = ::SSL_write_ex(this->ssl(), data + nskip, size - nskip, &datalen);

      if(ret == 0) {
        ssl_err = ::SSL_get_error(this->ssl(), ret);

        if((ssl_err == SSL_ERROR_WANT_READ) || (ssl_err == SSL_ERROR_WANT_WRITE))
          break;

        POSEIDON_LOG_ERROR((
            "Error writing SSL socket",
            "[`SSL_write_ex()` failed: SSL_get_error = `$3`, ERR_peek_error = `$4`, errno = `$5`]",
            "[SSL socket `$1` (class `$2`)]"),
            this, typeid(*this), ssl_err, ::ERR_reason_error_string(::ERR_peek_error()), format_errno());

        // The connection is now broken.
        this->quick_shut_down();
        return false;
      }

      // Discard data that have been sent.
      nskip += datalen;
    }

    // If the operation has completed only partially, buffer remaining data.
    // Space has already been reserved so this will not throw exceptions.
    ::memcpy(queue.mut_end(), data + nskip, size - nskip);
    queue.accept(size - nskip);
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
ssl_send_oob(char data) noexcept
  {
    return ::send(this->fd(), &data, 1, MSG_OOB) > 0;
  }

bool
SSL_Socket::
ssl_shut_down() noexcept
  {
    recursive_mutex::unique_lock io_lock;
    auto& queue = this->do_abstract_socket_lock_write_queue(io_lock);

    // If there are data pending, mark this socket as being closed. If a full
    // connection has been established, wait until all pending data to be sent.
    // The connection should be closed thereafter.
    if(!queue.empty() && this->do_abstract_socket_set_state(socket_state_established, socket_state_closing))
      return true;

    // If there are no data pending, shut it down immediately.
    ::SSL_shutdown(this->ssl());
    return ::shutdown(this->fd(), SHUT_RDWR) == 0;
  }

}  // namespace poseidon
