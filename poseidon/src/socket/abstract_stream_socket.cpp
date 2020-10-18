// This file is part of Poseidon.
// Copyleft 2020, LH_Mouse. All wrongs reserved.

#include "../precompiled.hpp"
#include "abstract_stream_socket.hpp"
#include "../static/network_driver.hpp"
#include "../util.hpp"

namespace poseidon {

Abstract_Stream_Socket::
~Abstract_Stream_Socket()
  {
  }

IO_Result
Abstract_Stream_Socket::
do_call_stream_preshutdown_unlocked()
  noexcept
  try {
    // Call `do_stream_preshutdown_unlocked()`, ignoring any exeptions.
    return this->do_stream_preshutdown_unlocked();
  }
  catch(exception& stdex) {
    POSEIDON_LOG_WARN("Failed to perform graceful shutdown on socket `$1`:\n"
                      "$2\n"
                      "[socket type `$3`]",
                      this, stdex, typeid(*this));
    return io_result_eof;
  }

IO_Result
Abstract_Stream_Socket::
do_async_shutdown_unlocked()
  noexcept
  {
    switch(this->m_cstate) {
      case connection_state_initial:
      case connection_state_connecting:
        // Shut down the connection. Discard pending data.
        ::shutdown(this->get_fd(), SHUT_RDWR);
        this->m_cstate = connection_state_closed;
        POSEIDON_LOG_TRACE("Marked socket `$1` as CLOSED (not established)", this);
        return io_result_eof;

      case connection_state_established: {
        // Ensure pending data are delivered.
        if(this->m_wqueue.size()) {
          ::shutdown(this->get_fd(), SHUT_RD);
          this->m_cstate = connection_state_closing;
          POSEIDON_LOG_TRACE("Marked socket `$1` as CLOSING (data pending)", this);
          return io_result_not_eof;
        }

        // Wait for shutdown.
        auto io_res = this->do_call_stream_preshutdown_unlocked();
        if(io_res != io_result_eof) {
          ::shutdown(this->get_fd(), SHUT_RD);
          this->m_cstate = connection_state_closing;
          POSEIDON_LOG_TRACE("Marked socket `$1` as CLOSING (preshutdown pending)", this);
          return io_res;
        }

        // Close the connection.
        ::shutdown(this->get_fd(), SHUT_RDWR);
        this->m_cstate = connection_state_closed;
        POSEIDON_LOG_TRACE("Marked socket `$1` as CLOSED (pending data clear)", this);
        return io_result_eof;
      }

      case connection_state_closing: {
        // Ensure pending data are delivered.
        if(this->m_wqueue.size())
          return io_result_not_eof;

        // Wait for shutdown.
        auto io_res = this->do_call_stream_preshutdown_unlocked();
        if(io_res != io_result_eof)
          return io_res;

        // Close the connection.
        ::shutdown(this->get_fd(), SHUT_RDWR);
        this->m_cstate = connection_state_closed;
        POSEIDON_LOG_TRACE("Marked socket `$1` as CLOSED (pending data clear)", this);
        return io_result_eof;
      }

      case connection_state_closed:
        // Do nothing.
        return io_result_eof;

      default:
        ROCKET_ASSERT(false);
    }
  }

IO_Result
Abstract_Stream_Socket::
do_on_async_poll_read(simple_mutex::unique_lock& lock, void* hint, size_t size)
  {
    lock.lock(this->m_io_mutex);

    // If the stream is in CLOSED state, fail.
    if(this->m_cstate == connection_state_closed)
      return io_result_eof;

    // Try reading some bytes.
    auto io_res = this->do_stream_read_unlocked(hint, size);
    if(io_res < 0)
      return io_res;

    if(io_res == io_result_eof) {
      POSEIDON_LOG_TRACE("End of stream encountered: $1", this);
      this->do_async_shutdown_unlocked();
      return io_result_eof;
    }

    // Process the data that have been read.
    lock.unlock();
    this->do_on_async_receive(hint, static_cast<size_t>(io_res));
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
    size_t size = this->m_wqueue.size();
    if(size != 0)
      return size;

    // If a shutdown request is pending, report at least one byte.
    if(this->m_cstate == connection_state_closing)
      return 1;

    // There is nothing to write.
    return 0;
  }

IO_Result
Abstract_Stream_Socket::
do_on_async_poll_write(simple_mutex::unique_lock& lock, void* /*hint*/, size_t /*size*/)
  {
    lock.lock(this->m_io_mutex);

    // If the stream is in CLOSED state, fail.
    if(this->m_cstate == connection_state_closed)
      return io_result_eof;

    // If the stream is in CONNECTING state, mark it ESTABLISHED.
    if(this->m_cstate < connection_state_established) {
      this->m_cstate = connection_state_established;

      lock.unlock();
      this->do_on_async_establish();
      lock.lock(this->m_io_mutex);
    }

    // Try writing some bytes.
    size_t size = this->m_wqueue.size();
    if(size == 0) {
      if(this->m_cstate <= connection_state_established)
        return io_result_eof;

      // Shut down the connection completely now.
      return this->do_async_shutdown_unlocked();
    }

    auto io_res = this->do_stream_write_unlocked(this->m_wqueue.data(), size);
    if(io_res < 0)
      return io_res;

    // Remove data that have been written.
    this->m_wqueue.discard(static_cast<size_t>(io_res));
    return io_res;
  }

void
Abstract_Stream_Socket::
do_on_async_poll_shutdown(int err)
  {
    simple_mutex::unique_lock lock(this->m_io_mutex);
    this->m_cstate = connection_state_closed;
    lock.unlock();

    this->do_on_async_shutdown(err);
  }

void
Abstract_Stream_Socket::
do_on_async_receive(void* data, size_t size)
  {
    // Append data to the default queue.
    // Note the queue is provided purely for convenience for derived classes. It is not
    // protected by the I/O mutex.
    this->m_rqueue.putn(static_cast<const char*>(data), size);

    this->do_on_async_receive(::std::move(this->m_rqueue));
  }

void
Abstract_Stream_Socket::
do_async_connect(const Socket_Address& addr)
  {
    // Lock the stream and examine connection state.
    simple_mutex::unique_lock lock(this->m_io_mutex);
    if(this->m_cstate != connection_state_initial)
      POSEIDON_THROW("Another connection already in progress or established");

    // Initiate the connection.
    this->do_stream_preconnect_unlocked();

    // No matter whether `::connect()` succeeds or fails with `EINPROGRESS`, the current
    // socket is set to the CONNECTING state.
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
do_async_send(const void* data, size_t size)
  {
    simple_mutex::unique_lock lock(this->m_io_mutex);
    if(this->m_cstate > connection_state_established)
      return false;

    // Append data to the write queue.
    this->m_wqueue.putn(static_cast<const char*>(data), size);
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

        // Cache the result.
        this->m_remote_addr.assign(addrst, addrlen);
      });

    return this->m_remote_addr;
  }

bool
Abstract_Stream_Socket::
shut_down()
  noexcept
  {
    simple_mutex::unique_lock lock(this->m_io_mutex);
    if(this->m_cstate > connection_state_established)
      return false;

    // Initiate asynchronous shutdown.
    this->do_async_shutdown_unlocked();
    lock.unlock();

    // Notify the driver about availability of outgoing data.
    Network_Driver::notify_writable_internal(this);
    return true;
  }

}  // namespace poseidon
