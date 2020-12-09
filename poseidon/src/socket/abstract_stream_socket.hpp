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
    mutable simple_mutex m_io_mutex;
    Connection_State m_cstate = connection_state_empty;
    linear_buffer m_wqueue;  // write queue

    linear_buffer m_rqueue;  // default read queue (not protected by I/O mutex)

    // This the remote address. It is initialized upon the first request.
    mutable once_flag m_remote_addr_once;
    mutable Socket_Address m_remote_addr;

  protected:
    // Adopts a foreign or accepted socket.
    explicit
    Abstract_Stream_Socket(unique_FD&& fd);

    // Creates a new non-blocking socket.
    explicit
    Abstract_Stream_Socket(::sa_family_t family);

  private:
    inline
    IO_Result
    do_socket_close_unlocked()
      noexcept;

    // Reads some data.
    // `lock` will lock `*this` after the call.
    // `hint` is used as the I/O buffer. `size` specifies the maximum number of
    // bytes to read.
    IO_Result
    do_socket_on_poll_read(simple_mutex::unique_lock& lock, char* hint, size_t size)
      final;

    // Returns the estimated size of data pending for writing.
    // `lock` will lock `*this` after the call.
    size_t
    do_write_queue_size(simple_mutex::unique_lock& lock)
      const final;

    // Writes some data.
    // `lock` will lock `*this` after the call.
    // `hint` is ignored. `size` specifies the maximum number of bytes to write.
    IO_Result
    do_socket_on_poll_write(simple_mutex::unique_lock& lock, char* hint, size_t size)
      final;

    // Notifies a full-duplex channel has been closed.
    void
    do_socket_on_poll_close(int err)
      final;

  protected:
    // Performs read operation. Overridden functions shall update `data` to denote
    // the end of bytes that have been read.
    // This function is called by the network thread. The current socket will have
    // been locked by its caller. No synchronization is required.
    virtual
    IO_Result
    do_socket_stream_read_unlocked(char*& data, size_t size)
      = 0;

    // Performs write operation. Overridden functions shall update `data` to denote
    // the end of bytes that have been written.
    // This function is called by the network thread. The current socket will have
    // been locked by its caller. No synchronization is required.
    virtual
    IO_Result
    do_socket_stream_write_unlocked(const char*& data, size_t size)
      = 0;

    // Notifies a full-duplex channel has been established.
    // Please mind thread safety, as this function is called by the network thread.
    virtual
    void
    do_socket_on_establish()
      = 0;

    // Consumes an incoming packet.
    // Please mind thread safety, as this function is called by the network thread.
    // The overload taking a pointer and a size appends data to `m_rqueue`, then
    // calls the other overload. Derived classes may override either (but not both)
    // overload for convenience.
    virtual
    void
    do_socket_on_receive(char* data, size_t size);

    virtual
    void
    do_socket_on_receive(linear_buffer&& rqueue)
      = 0;

    // Notifies a socket has been closed.
    // Please mind thread safety, as this function is called by the network thread.
    virtual
    void
    do_socket_on_close(int err)
      = 0;

    // Initiates a new connection to the specified address.
    void
    do_socket_connect(const Socket_Address& addr);

    // Enqueues some data for writing.
    // This function returns `true` if the data have been queued, or `false` if a
    // shutdown request has been initiated.
    // This function is thread-safe.
    bool
    do_socket_send(const char* data, size_t size);

  public:
    ASTERIA_NONCOPYABLE_DESTRUCTOR(Abstract_Stream_Socket);

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
    close()
      noexcept;
  };

}  // namespace poseidon

#endif
