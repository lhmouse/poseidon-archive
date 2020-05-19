// This file is part of Poseidon.
// Copyleft 2020, LH_Mouse. All wrongs reserved.

#include "../precompiled.hpp"
#include "abstract_tcp_socket.hpp"
#include "socket_address.hpp"
#include "../utilities.hpp"
#include <netinet/in.h>
#include <netinet/tcp.h>

namespace poseidon {
namespace {

IO_Result
do_translate_syscall_error(const char* func, int err)
  {
    if(err == EINTR)
      return io_result_intr;

    if(::rocket::is_any_of(err, { EAGAIN, EWOULDBLOCK }))
      return io_result_again;

    POSEIDON_THROW("TCP socket error\n"
                   "[`$1()` failed: $2]",
                   func, noadl::format_errno(err));
  }

}  // namespace

Abstract_TCP_Socket::
~Abstract_TCP_Socket()
  {
  }

void
Abstract_TCP_Socket::
do_set_common_options()
  {
    // Disables Nagle algorithm.
    static constexpr int true_val[] = { -1 };
    int res = ::setsockopt(this->get_fd(), IPPROTO_TCP, TCP_NODELAY,
                           true_val, sizeof(true_val));
    ROCKET_ASSERT(res == 0);
  }

void
Abstract_TCP_Socket::
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
          this->m_cstate = connection_state_closing;
          POSEIDON_LOG_TRACE("Marked TCP socket as CLOSING (data pending): $1", this);
          break;
        }

        // Shut down the connection if there are no pending data.
        ::shutdown(this->get_fd(), SHUT_RDWR);
        this->m_cstate = connection_state_closed;
        POSEIDON_LOG_TRACE("Marked TCP socket as CLOSED (no data pending): $1", this);
        break;

      case connection_state_closing:
      case connection_state_closed:
        // Do nothing.
        break;
    }
  }

IO_Result
Abstract_TCP_Socket::
do_on_async_poll_read(Rc_Mutex::unique_lock& lock, void* hint, size_t size)
  {
    // Lock the stream before performing other operations.
    lock.assign(this->m_mutex);

    // Try reading some bytes.
    ::ssize_t nread = ::read(this->get_fd(), hint, size);
    if(nread > 0) {
      this->do_on_async_receive(hint, static_cast<size_t>(nread));
      return static_cast<IO_Result>(nread);
    }

    // Shut down the connection if FIN received.
    if(nread == 0)
      this->do_async_shutdown_nolock();

    return do_translate_syscall_error("read", errno);
  }

size_t
Abstract_TCP_Socket::
do_write_queue_size(Rc_Mutex::unique_lock& lock)
const
  {
    // Lock the stream before performing other operations.
    lock.assign(this->m_mutex);

    // Get the size of pending data.
    return this->m_wqueue.size();
  }

IO_Result
Abstract_TCP_Socket::
do_on_async_poll_write(Rc_Mutex::unique_lock& lock, void* /*hint*/, size_t /*size*/)
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

      // Shut down the connection completely now.
      ::shutdown(this->get_fd(), SHUT_RDWR);
      this->m_cstate = connection_state_closed;
      POSEIDON_LOG_TRACE("Marked TCP socket as CLOSED (pending data cleared): $1", this);
      return io_result_eof;
    }

    // Try writing some bytes.
    ::ssize_t nwritten = ::write(this->get_fd(), this->m_wqueue.data(), size);
    if(nwritten > 0) {
      this->m_wqueue.discard(static_cast<size_t>(nwritten));
      return static_cast<IO_Result>(nwritten);
    }

    return do_translate_syscall_error("write", errno);
  }

void
Abstract_TCP_Socket::
do_on_async_establish()
  {
    POSEIDON_LOG_INFO("TCP connection established: local '$1', remote '$2'",
                      this->get_local_address(), this->get_remote_address());
  }

void
Abstract_TCP_Socket::
do_on_async_poll_shutdown(int err)
  {
    POSEIDON_LOG_INFO("TCP connection closed: local '$1', reason: $2",
                      this->get_local_address(), noadl::format_errno(err));
  }

void
Abstract_TCP_Socket::
async_connect(const Socket_Address& addr)
  {
    Rc_Mutex::unique_lock lock(this->m_mutex);
    if(this->m_cstate != connection_state_initial)
      POSEIDON_THROW("another connection already in progress");

    // Initiate the connection.
    // Whether `::connect()` succeeds or fails with `EINPROGRESS`, the current
    // socket is set to the CONNECTING state.
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
Abstract_TCP_Socket::
async_send(const void* data, size_t size)
  {
    Rc_Mutex::unique_lock lock(this->m_mutex);
    if(this->m_cstate > connection_state_established)
      return false;

    // Append data to the write queue.
    this->m_wqueue.putn(static_cast<const char*>(data), size);
    lock.unlock();

    // TODO notify network driver
    return true;
  }

void
Abstract_TCP_Socket::
async_shutdown()
noexcept
  {
    Rc_Mutex::unique_lock lock(this->m_mutex);
    this->do_async_shutdown_nolock();
  }

}  // namespace poseidon
