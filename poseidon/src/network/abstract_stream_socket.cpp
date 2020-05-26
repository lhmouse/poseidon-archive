// This file is part of Poseidon.
// Copyleft 2020, LH_Mouse. All wrongs reserved.

#include "../precompiled.hpp"
#include "abstract_stream_socket.hpp"
#include "socket_address.hpp"
#include "../static/network_driver.hpp"
#include "../utilities.hpp"

namespace poseidon {

Abstract_Stream_Socket::
~Abstract_Stream_Socket()
  {
  }

IO_Result
Abstract_Stream_Socket::
do_call_stream_preshutdown_nolock()
noexcept
  {
    // Call `do_stream_preshutdown_nolock()`, ignoring any exeptions.
    auto io_res = io_result_eof;
    try {
      io_res = this->do_stream_preshutdown_nolock();
    }
    catch(const exception& stdex) {
      POSEIDON_LOG_WARN("Failed to perform graceful shutdown on stream socket: $1", this);
    }
    return io_res;
  }

void
Abstract_Stream_Socket::
do_async_shutdown_nolock()
noexcept
  {
    switch(this->m_cstate) {
      case connection_state_initial:
      case connection_state_connecting:
        // Shut down the connection. Discard pending data.
        ::shutdown(this->get_fd(), SHUT_RDWR);
        this->m_cstate = connection_state_closed;
        POSEIDON_LOG_TRACE("Marked stream socket as CLOSED (not established): $1", this);
        break;

      case connection_state_established:
      case connection_state_closing:
        // Ensure pending data are delivered.
        if(this->m_wqueue.size() || (this->do_call_stream_preshutdown_nolock() != io_result_eof)) {
          ::shutdown(this->get_fd(), SHUT_RD);
          this->m_cstate = connection_state_closing;
          POSEIDON_LOG_TRACE("Marked stream socket as CLOSING (data pending): $1", this);
        }
        else {
          ::shutdown(this->get_fd(), SHUT_RDWR);
          this->m_cstate = connection_state_closed;
          POSEIDON_LOG_TRACE("Marked stream socket as CLOSED (no data pending): $1", this);
        }
        break;

      case connection_state_closed:
        // Do nothing.
        break;
    }
  }

IO_Result
Abstract_Stream_Socket::
do_on_async_poll_read(Si_Mutex::unique_lock& lock, void* hint, size_t size)
  {
    lock.assign(this->m_mutex);

    // If the stream is in CLOSED state, fail.
    if(this->m_cstate == connection_state_closed)
      return io_result_eof;

    // Try reading some bytes.
    auto io_res = this->do_stream_read_nolock(hint, size);
    if(io_res < 0)
      return io_res;

    if(io_res == io_result_eof) {
      this->do_async_shutdown_nolock();
      return io_res;
    }

    // Process the data that have been read.
    lock.unlock();
    this->do_on_async_receive(hint, static_cast<size_t>(io_res));
    lock.assign(this->m_mutex);
    return io_res;
  }

size_t
Abstract_Stream_Socket::
do_write_queue_size(Si_Mutex::unique_lock& lock)
const
  {
    lock.assign(this->m_mutex);

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
do_on_async_poll_write(Si_Mutex::unique_lock& lock, void* /*hint*/, size_t /*size*/)
  {
    lock.assign(this->m_mutex);

    // If the stream is in CLOSED state, fail.
    if(this->m_cstate == connection_state_closed)
      return io_result_eof;

    // If the stream is in CONNECTING state, mark it ESTABLISHED.
    if(this->m_cstate < connection_state_established) {
      this->m_cstate = connection_state_established;

      lock.unlock();
      this->do_on_async_establish();
      lock.assign(this->m_mutex);
    }

    if(this->m_cstate <= connection_state_established) {
      // Try writing some bytes.
      size_t size = this->m_wqueue.size();
      if(size == 0)
        return io_result_eof;

      auto io_res = this->do_stream_write_nolock(this->m_wqueue.data(), size);
      if(io_res < 0)
        return io_res;

      // Remove data that have been written.
      this->m_wqueue.discard(static_cast<size_t>(io_res));
      return io_res;
    }

    // Shut down the connection completely now.
    auto io_res = this->do_call_stream_preshutdown_nolock();
    if(io_res != io_result_eof)
      return io_res;

    // Shutdown complete. Close thw connection now.
    ::shutdown(this->get_fd(), SHUT_RDWR);
    this->m_cstate = connection_state_closed;
    POSEIDON_LOG_TRACE("Marked stream socket as CLOSED (pending data clear): $1", this);
    return io_res;
  }

void
Abstract_Stream_Socket::
do_on_async_poll_shutdown(int err)
  {
    Si_Mutex::unique_lock lock(this->m_mutex);
    this->m_cstate = connection_state_closed;
    lock.unlock();

    this->do_on_async_shutdown(err);
  }

void
Abstract_Stream_Socket::
do_async_connect(const Socket_Address& addr)
  {
    // Lock the stream and examine connection state.
    Si_Mutex::unique_lock lock(this->m_mutex);
    if(this->m_cstate != connection_state_initial)
      POSEIDON_THROW("another connection already in progress or established");

    // Initiate the connection.
    this->do_stream_preconnect_nolock();

    // No matter whether `::connect()` succeeds or fails with `EINPROGRESS`, the current
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
Abstract_Stream_Socket::
async_send(const void* data, size_t size)
  {
    Si_Mutex::unique_lock lock(this->m_mutex);
    if(this->m_cstate > connection_state_established)
      return false;

    // Append data to the write queue.
    this->m_wqueue.putn(static_cast<const char*>(data), size);
    lock.unlock();

    // Notify the driver about availability of outgoing data.
    Network_Driver::notify_writable_internal(this);
    return true;
  }

bool
Abstract_Stream_Socket::
async_shutdown()
noexcept
  {
    Si_Mutex::unique_lock lock(this->m_mutex);
    if(this->m_cstate == connection_state_closed)
      return false;

    // Initiate asynchronous shutdown.
    this->do_async_shutdown_nolock();
    lock.unlock();

    // Notify the driver about availability of outgoing data.
    Network_Driver::notify_writable_internal(this);
    return true;
  }

}  // namespace poseidon
