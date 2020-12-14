// This file is part of Poseidon.
// Copyleft 2020, LH_Mouse. All wrongs reserved.

#include "../precompiled.hpp"
#include "abstract_stream_socket.hpp"
#include "../static/network_driver.hpp"
#include "../utils.hpp"
#include <netinet/tcp.h>

namespace poseidon {

Abstract_Stream_Socket::
Abstract_Stream_Socket(unique_FD&& fd)
  : Abstract_Socket(::std::move(fd))
  {
  }

Abstract_Stream_Socket::
Abstract_Stream_Socket(::sa_family_t family)
  : Abstract_Socket(family, SOCK_STREAM, IPPROTO_TCP)
  {
  }

Abstract_Stream_Socket::
~Abstract_Stream_Socket()
  {
  }

IO_Result
Abstract_Stream_Socket::
do_socket_close_unlocked()
  noexcept
  {
    switch(this->m_cstate) {
      case connection_state_empty:
      case connection_state_connecting:
        // Shut down the connection. Discard pending data.
        this->m_cstate = connection_state_closed;
        ::shutdown(this->get_fd(), SHUT_RDWR);
        POSEIDON_LOG_TRACE("Marked socket `$1` as CLOSED (not established)", this);
        return io_result_end_of_stream;

      case connection_state_established: {
        // Ensure pending data are delivered.
        this->m_cstate = connection_state_closing;
        ::shutdown(this->get_fd(), SHUT_RD);

        // Fallthrough
      case connection_state_closing:
        if(this->m_wqueue.size()) {
          POSEIDON_LOG_TRACE("Marked socket `$1` as CLOSING (data pending)", this);
          return io_result_partial_work;
        }

        // For TLS streams, this sends the 'close notify' alert.
        this->do_socket_stream_preclose_unclocked();

        // Close the connection completely.
        this->m_cstate = connection_state_closed;
        ::shutdown(this->get_fd(), SHUT_RDWR);
        POSEIDON_LOG_TRACE("Marked socket `$1` as CLOSED (data clear)", this);
        return io_result_end_of_stream;
      }

      case connection_state_closed:
        // Do nothing.
        return io_result_end_of_stream;

      default:
        ROCKET_ASSERT(false);
    }
  }

IO_Result
Abstract_Stream_Socket::
do_socket_on_poll_read(simple_mutex::unique_lock& lock, char* hint, size_t size)
  {
    ROCKET_ASSERT(size != 0);
    lock.lock(this->m_io_mutex);
    if(this->m_cstate == connection_state_closed)
      return io_result_end_of_stream;

    // Try reading some bytes.
    char* eptr = hint;
    auto io_res = this->do_socket_stream_read_unlocked(eptr, size);
    if(io_res == io_result_end_of_stream) {
      POSEIDON_LOG_TRACE("End of stream encountered: $1", this);
      this->do_socket_close_unlocked();
    }
    if(io_res != io_result_partial_work)
      return io_res;

    // Process the data that have been read.
    lock.unlock();
    this->do_socket_on_receive(hint, static_cast<size_t>(eptr - hint));

    lock.lock(this->m_io_mutex);
    return io_res;
  }

size_t
Abstract_Stream_Socket::
do_write_queue_size(simple_mutex::unique_lock& lock)
  const
  {
    lock.lock(this->m_io_mutex);

    // Get the size of pending data.
    size_t navail = this->m_wqueue.size();
    if(navail != 0)
      return navail;

    // If a shutdown request is pending, report at least one byte.
    return this->m_cstate == connection_state_closing;
  }

IO_Result
Abstract_Stream_Socket::
do_socket_on_poll_write(simple_mutex::unique_lock& lock, char* /*hint*/, size_t size)
  {
    ROCKET_ASSERT(size != 0);
    lock.lock(this->m_io_mutex);
    if(this->m_cstate == connection_state_closed)
      return io_result_end_of_stream;

    // If the stream is in CONNECTING state, mark it ESTABLISHED.
    if(this->m_cstate < connection_state_established) {
      this->m_cstate = connection_state_established;

      // Disables Nagle algorithm. Errors are ignored.
      static constexpr int yes[] = { -1 };
      ::setsockopt(this->get_fd(), IPPROTO_TCP, TCP_NODELAY, yes, sizeof(yes));

      lock.unlock();
      this->do_socket_on_establish();
    }
    lock.lock(this->m_io_mutex);

    // Try writing some bytes.
    size_t navail = ::std::min(this->m_wqueue.size(), size);
    if((navail == 0) && (this->m_cstate > connection_state_established))
      return this->do_socket_close_unlocked();

    if(navail == 0)
      return io_result_end_of_stream;

    const char* eptr = this->m_wqueue.data();
    auto io_res = this->do_socket_stream_write_unlocked(eptr, navail);
    if(io_res != io_result_partial_work)
      return io_res;

    // Remove data that have been written.
    this->m_wqueue.discard(static_cast<size_t>(eptr - this->m_wqueue.data()));
    return io_res;
  }

void
Abstract_Stream_Socket::
do_socket_on_poll_close(int err)
  {
    simple_mutex::unique_lock lock(this->m_io_mutex);
    this->m_cstate = connection_state_closed;
    lock.unlock();

    this->do_socket_on_close(err);
  }

void
Abstract_Stream_Socket::
do_socket_on_receive(char* data, size_t size)
  {
    // Append data to the default queue.
    // Note the queue is provided purely for convenience for derived classes.
    // It is not protected by the I/O mutex.
    this->m_rqueue.putn(data, size);

    this->do_socket_on_receive(::std::move(this->m_rqueue));
  }

void
Abstract_Stream_Socket::
do_socket_connect(const Socket_Address& addr)
  {
    // Lock the stream and examine connection state.
    simple_mutex::unique_lock lock(this->m_io_mutex);
    if(this->m_cstate != connection_state_empty)
      POSEIDON_THROW("Another connection already in progress or established");

    // No matter whether `::connect()` succeeds or fails with `EINPROGRESS`,
    // the current socket is set to the CONNECTING state.
    if(::connect(this->get_fd(), addr.data(), addr.ssize()) != 0) {
      int err = errno;
      if(err != EINPROGRESS)
        POSEIDON_THROW("Failed to initiate connection to '$2'\n"
                       "[`connect()` failed: $1]",
                       format_errno(err), addr);
    }
    this->m_cstate = connection_state_connecting;
  }

bool
Abstract_Stream_Socket::
do_socket_send(const char* data, size_t size)
  {
    simple_mutex::unique_lock lock(this->m_io_mutex);
    if(this->m_cstate > connection_state_established)
      return false;

    // Append data to the write queue.
    this->m_wqueue.putn(data, size);
    lock.unlock();

    // Notify the driver about availability of outgoing data.
    Network_Driver::notify_writable_internal(this);
    return true;
  }

const Socket_Address&
Abstract_Stream_Socket::
get_remote_address()
  const
  {
    this->m_remote_addr_once.call(
      [this] {
        // Try getting the remote address.
        Socket_Address::storage addrst;
        ::socklen_t addrlen = sizeof(addrst);
        if(::getsockname(this->get_fd(), addrst, &addrlen) != 0)
          POSEIDON_THROW("Could not get remote socket address\n"
                         "[`getsockname()` failed: $1]",
                         format_errno(errno));

        // The result is cached once it becomes available.
        this->m_remote_addr.assign(addrst, addrlen);
      });
    return this->m_remote_addr;
  }

bool
Abstract_Stream_Socket::
close()
  noexcept
  {
    simple_mutex::unique_lock lock(this->m_io_mutex);
    if(this->m_cstate > connection_state_established)
      return false;

    // Initiate asynchronous shutdown.
    this->do_socket_close_unlocked();
    lock.unlock();

    // Notify the driver about availability of outgoing data.
    Network_Driver::notify_writable_internal(this);
    return true;
  }

}  // namespace poseidon
