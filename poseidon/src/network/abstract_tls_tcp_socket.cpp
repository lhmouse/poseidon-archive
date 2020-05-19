// This file is part of Poseidon.
// Copyleft 2020, LH_Mouse. All wrongs reserved.

#include "../precompiled.hpp"
#include "abstract_tls_tcp_socket.hpp"
#include "socket_address.hpp"
#include "../utilities.hpp"
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <openssl/err.h>

namespace poseidon {
namespace {

ROCKET_NOINLINE
size_t
do_print_errors()
  {
    char sbuf[1024];
    size_t index = 0;

    while(unsigned long err = ::ERR_get_error()) {
      ::ERR_error_string_n(err, sbuf, sizeof(sbuf));
      POSEIDON_LOG_ERROR("OpenSSL error: [$1] $2", index, err);
      ++index;
    }
    return index;
  }

IO_Result
do_translate_ssl_error(const char* func, ::SSL* ssl, int ret)
  {
    int err = ::SSL_get_error(ssl, ret);
    switch(err) {
      case SSL_ERROR_NONE:
      case SSL_ERROR_ZERO_RETURN:
        // normal closure
        return io_result_eof;

      case SSL_ERROR_WANT_READ:
      case SSL_ERROR_WANT_WRITE:
      case SSL_ERROR_WANT_CONNECT:
      case SSL_ERROR_WANT_ACCEPT:
        // retry
        return io_result_again;

      case SSL_ERROR_SYSCALL:
        // get syscall errno
        err = errno;

        // dump and clear the thread-local error queue
        do_print_errors();

        // forward `errno` verbatim
        if(err == EINTR)
          return io_result_intr;

        POSEIDON_THROW("OpenSSL reported an irrecoverable I/O error\n"
                       "[`$1()` returned `$2`; errno: $3]",
                       func, ret, noadl::format_errno(errno));

      default:
        // dump and clear the thread-local error queue
        do_print_errors();

        // report generic failure
        POSEIDON_THROW("OpenSSL reported an irrecoverable error\n"
                       "[`$1()` returned `$2`]",
                       func, ret);
    }
  }

}  // namespace

Abstract_TLS_TCP_Socket::
~Abstract_TLS_TCP_Socket()
  {
  }

void
Abstract_TLS_TCP_Socket::
do_set_common_options()
  {
    // Disables Nagle algorithm.
    static constexpr int true_val[] = { -1 };
    int res = ::setsockopt(this->get_fd(), IPPROTO_TCP, TCP_NODELAY,
                           true_val, sizeof(true_val));
    ROCKET_ASSERT(res == 0);

    // Create BIO.
    res = ::SSL_set_fd(this->m_ssl, this->get_fd());
    if(res == 0) {
      // The OpenSSL documentation says errors are to be retrieved from
      // 'the error stack'... where is it?
      do_print_errors();

      POSEIDON_THROW("could not set OpenSSL file descriptor\n",
                     "[`SSL_set_fd()` returned `$1`]",
                     res);
    }

    // Set default SSL mode. This will be changed if `connect()` is called.
    ::SSL_set_accept_state(this->m_ssl);
  }

void
Abstract_TLS_TCP_Socket::
do_async_shutdown_nolock()
noexcept
  {
    switch(this->m_cstate) {
      case connection_state_initial:
      case connection_state_connecting:
        // Shut down the connection. Discard pending data.
        ::shutdown(this->get_fd(), SHUT_RDWR);
        this->m_cstate = connection_state_closed;
        break;

      case connection_state_established:
        // Shut down the read part.
        if(this->m_wqueue.size()) {
          ::shutdown(this->get_fd(), SHUT_RD);
          ::SSL_shutdown(this->m_ssl);
          this->m_cstate = connection_state_closing;
          POSEIDON_LOG_TRACE("Marked secure TCP socket as CLOSING (data pending): $1", this);
          break;
        }

        // Shut down the connection if there are no pending data.
        ::shutdown(this->get_fd(), SHUT_RDWR);
        this->m_cstate = connection_state_closed;
        POSEIDON_LOG_TRACE("Marked secure TCP socket as CLOSED (no data pending): $1", this);
        break;

      case connection_state_closing:
      case connection_state_closed:
        // Do nothing.
        break;
    }
  }

IO_Result
Abstract_TLS_TCP_Socket::
do_on_async_read(Rc_Mutex::unique_lock& lock, void* hint, size_t size)
  {
    // Lock the stream before performing other operations.
    lock.assign(this->m_mutex);

    // Try reading some bytes.
    // Note: According to OpenSSL documentation, unlike `::read()`, a return value
    //       of zero indates failure, rather than EOF.
    int nread = ::SSL_read(this->m_ssl, hint,
                           static_cast<int>(::rocket::min(size, UINT_MAX / 2)));
    if(nread > 0) {
      this->do_on_async_receive(hint, static_cast<size_t>(nread));
      return static_cast<IO_Result>(nread);
    }

    // Shut down the connection if FIN received.
    if(::SSL_get_shutdown(this->m_ssl) & SSL_RECEIVED_SHUTDOWN)
       this->do_async_shutdown_nolock();

    return do_translate_ssl_error("SSL_read", this->m_ssl, nread);
  }

size_t
Abstract_TLS_TCP_Socket::
do_write_queue_size(Rc_Mutex::unique_lock& lock)
const
  {
    // Lock the stream before performing other operations.
    lock.assign(this->m_mutex);

    // Get the size of pending data.
    return this->m_wqueue.size();
  }

IO_Result
Abstract_TLS_TCP_Socket::
do_on_async_write(Rc_Mutex::unique_lock& lock, void* /*hint*/, size_t /*size*/)
  {
    // Lock the stream before performing other operations.
    lock.assign(this->m_mutex);

    // Mark the connection fully established.
    if(this->m_cstate < connection_state_established) {
      this->m_cstate = connection_state_established;
      this->do_on_async_establish();
    }

    // If the write queue is empty, there is nothing to do.
    size_t size = this->m_wqueue.size();
    if(size == 0) {
      // Check for pending shutdown requests.
      if(this->m_cstate != connection_state_closing)
        return io_result_eof;

      // Request bidirectional shutdown.
      int ret = ::SSL_shutdown(this->m_ssl);
      if(ret == 0)
        ret = ::SSL_shutdown(this->m_ssl);

      auto io_res = do_translate_ssl_error("SSL_shutdown", this->m_ssl, ret);
      if(io_res != io_result_eof)
        return io_res;

      // Shut down the connection completely now.
      ::shutdown(this->get_fd(), SHUT_RDWR);
      this->m_cstate = connection_state_closed;
      POSEIDON_LOG_TRACE("Marked secure TCP socket as CLOSED (pending data cleared): $1", this);
      return io_result_eof;
    }

    // Try writing some bytes.
    int nwritten = ::SSL_write(this->m_ssl, this->m_wqueue.data(),
                               static_cast<int>(::rocket::min(size, UINT_MAX / 2)));
    if(nwritten > 0) {
      this->m_wqueue.discard(static_cast<size_t>(nwritten));
      return static_cast<IO_Result>(nwritten);
    }

    return do_translate_ssl_error("SSL_write", this->m_ssl, nwritten);
  }

void
Abstract_TLS_TCP_Socket::
do_on_async_establish()
  {
    POSEIDON_LOG_INFO("Secure TCP connection established: local '$1', remote '$2'",
                      this->get_local_address(), this->get_remote_address());
  }

void
Abstract_TLS_TCP_Socket::
do_on_async_shutdown(int err)
  {
    POSEIDON_LOG_INFO("Secure TCP connection closed: local '$1', reason: $2",
                      this->get_local_address(), noadl::format_errno(err));
  }

void
Abstract_TLS_TCP_Socket::
async_connect(const Socket_Address& addr)
  {
    Rc_Mutex::unique_lock lock(this->m_mutex);
    if(this->m_cstate != connection_state_initial)
      POSEIDON_THROW("another connection is already in progress");

    ::SSL_set_connect_state(this->m_ssl);

    if(::connect(this->get_fd(), addr.data(), addr.size()) != 0) {
      int err = errno;
      if(err != EINPROGRESS)
        POSEIDON_THROW("failed to initiate connection to '$2'\n",
                       "[`connect()` failed: $1]",
                       noadl::format_errno(err), addr);
    }

    this->m_cstate = connection_state_connecting;
  }

bool
Abstract_TLS_TCP_Socket::
async_send(const void* data, size_t size)
  {
    // Append data to the write queue.
    Rc_Mutex::unique_lock lock(this->m_mutex);
    if(this->m_cstate > connection_state_established)
      return false;

    this->m_wqueue.putn(static_cast<const char*>(data), size);
    lock.unlock();

    // TODO notify network driver
    return true;
  }

void
Abstract_TLS_TCP_Socket::
async_shutdown()
noexcept
  {
    Rc_Mutex::unique_lock lock(this->m_mutex);
    this->do_async_shutdown_nolock();
  }

}  // namespace poseidon
