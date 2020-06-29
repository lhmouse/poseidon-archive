// This file is part of Poseidon.
// Copyleft 2020, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_SOCKET_ABSTRACT_STREAM_SOCKET_HPP_
#define POSEIDON_SOCKET_ABSTRACT_STREAM_SOCKET_HPP_

#include "abstract_socket.hpp"

namespace poseidon {

class Abstract_Stream_Socket
  : public ::asteria::Rcfwd<Abstract_Stream_Socket>,
    public Abstract_Socket
  {
  private:
    // These are I/O components.
    mutable mutex m_io_mutex;
    Connection_State m_cstate = connection_state_initial;
    linear_buffer m_wqueue;  // write queue

    linear_buffer m_rqueue;  // default read queue (not protected by I/O mutex)

    // This the remote address. It is initialized upon the first request.
    mutable once_flag m_remote_addr_once;
    mutable Socket_Address m_remote_addr;

  public:
    explicit
    Abstract_Stream_Socket(unique_FD&& fd)
      : Abstract_Socket(::std::move(fd))
      { }

    ASTERIA_NONCOPYABLE_DESTRUCTOR(Abstract_Stream_Socket);

  private:
    inline
    IO_Result
    do_call_stream_preshutdown_unlocked()
    noexcept;

    inline
    IO_Result
    do_async_shutdown_unlocked()
    noexcept;

    // Reads some data.
    // `lock` will lock `*this` after the call.
    // `hint` is used as the I/O buffer. `size` specifies the maximum number of
    // bytes to read.
    IO_Result
    do_on_async_poll_read(mutex::unique_lock& lock, void* hint, size_t size)
    final;

    // Returns the estimated size of data pending for writing.
    // `lock` will lock `*this` after the call.
    size_t
    do_write_queue_size(mutex::unique_lock& lock)
    const final;

    // Writes some data.
    // `lock` will lock `*this` after the call.
    // `hint` and `size` are ignored.
    IO_Result
    do_on_async_poll_write(mutex::unique_lock& lock, void* hint, size_t size)
    final;

    // Notifies a full-duplex channel has been closed.
    void
    do_on_async_poll_shutdown(int err)
    final;

  protected:
    // Performs outgoing connecting preparation.
    // This function is called by the network thread. The current socket will have
    // been locked by its caller. No synchronization is required.
    // This function is not called on incoming connections.
    virtual
    void
    do_stream_preconnect_unlocked()
      = 0;

    // Performs read operation.
    // This function is called by the network thread. The current socket will have
    // been locked by its caller. No synchronization is required.
    virtual
    IO_Result
    do_stream_read_unlocked(void* data, size_t size)
      = 0;

    // Performs write operation.
    // This function is called by the network thread. The current socket will have
    // been locked by its caller. No synchronization is required.
    virtual
    IO_Result
    do_stream_write_unlocked(const void* data, size_t size)
      = 0;

    // Performs shutdown preparation.
    // This function is called by the network thread. The current socket will have
    // been locked by its caller. No synchronization is required.
    // This function shall return `io_result_eof` if the shutdown operation has
    // either completed or failed due to some irrecoverable errors.
    virtual
    IO_Result
    do_stream_preshutdown_unlocked()
      = 0;

    // Notifies a full-duplex channel has been established.
    // Please mind thread safety, as this function is called by the network thread.
    virtual
    void
    do_on_async_establish()
      = 0;

    // Consumes an incoming packet.
    // Please mind thread safety, as this function is called by the network thread.
    // The overload taking a pointer and a size appends data to `m_rqueue`, then
    // calls the other overload. Derived classes may override either (but not both)
    // overload for convenience.
    virtual
    void
    do_on_async_receive(void* data, size_t size);

    virtual
    void
    do_on_async_receive(linear_buffer&& rqueue)
      = 0;

    // Notifies a socket has been closed.
    // Please mind thread safety, as this function is called by the network thread.
    virtual
    void
    do_on_async_shutdown(int err)
      = 0;

    // Initiates a new connection to the specified address.
    void
    do_async_connect(const Socket_Address& addr);

    // Enqueues some data for writing.
    // This function returns `true` if the data have been queued, or `false` if a
    // shutdown request has been initiated.
    // This function is thread-safe.
    bool
    do_async_send(const void* data, size_t size);

  public:
    // Gets the (connected) address of the remote peer.
    // This function throws an exception if no peer has connected.
    const Socket_Address&
    get_remote_address()
    const;

    // Initiates normal closure of this stream.
    // This function returns `true` if the shutdown request completes immediately,
    // or `false` if there are still pending data. In either case, the socket is
    // considered closed. No further data may be read from or written through it.
    // Note half-closed connections are not supported.
    bool
    async_shutdown()
    noexcept;
  };

}  // namespace poseidon

#endif
