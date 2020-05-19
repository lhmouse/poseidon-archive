// This file is part of Poseidon.
// Copyleft 2020, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_NETWORK_ABSTRACT_TCP_SOCKET_HPP_
#define POSEIDON_NETWORK_ABSTRACT_TCP_SOCKET_HPP_

#include "abstract_socket.hpp"
#include <rocket/linear_buffer.hpp>

namespace poseidon {

class Abstract_TCP_Socket
  : public Abstract_Socket
  {
  public:
    using base_type = Abstract_Socket;

  private:
    mutable Rc_Mutex m_mutex;
    Connection_State m_cstate = connection_state_initial;
    ::rocket::linear_buffer m_wqueue;  // write queue

  public:
    explicit
    Abstract_TCP_Socket(unique_posix_fd&& fd)
      : base_type(::std::move(fd))
      { this->do_set_common_options();  }

    ASTERIA_NONCOPYABLE_DESTRUCTOR(Abstract_TCP_Socket);

  private:
    // Disables Nagle algorithm, etc.
    void
    do_set_common_options();

    inline
    void
    do_async_shutdown_nolock()
    noexcept;

    // Reads some data.
    // `lock` will lock `*this` after the call if locking is supported.
    // `hint` is used as the I/O buffer. `size` specifies the maximum number of
    // bytes to read.
    IO_Result
    do_on_async_read(Rc_Mutex::unique_lock& lock, void* hint, size_t size)
    final;

    // Returns the size of data pending for writing.
    // `lock` will lock `*this` after the call if locking is supported.
    size_t
    do_write_queue_size(Rc_Mutex::unique_lock& lock)
    const final;

    // Writes some data.
    // `lock` will lock `*this` after the call if locking is supported.
    // `hint` and `size` are ignored.
    IO_Result
    do_on_async_write(Rc_Mutex::unique_lock& lock, void* hint, size_t size)
    final;

  protected:
    // Notifies a full-duplex channel has been established.
    // The default implementation does nothing.
    // Please mind thread safety, as this function is called by the network thread.
    virtual
    void
    do_on_async_establish();

    // Consumes incoming data.
    // Please mind thread safety, as this function is called by the network thread.
    virtual
    void
    do_on_async_receive(void* data, size_t size)
      = 0;

    // Notifies a full-duplex channel has been closed.
    // The default implementation does nothing.
    // Please mind thread safety, as this function is called by the network thread.
    void
    do_on_async_shutdown(int err)
    override;

  public:
    // Initiates a new connection to the specified address.
    void
    async_connect(const Socket_Address& addr);

    // Enqueues some data for writing.
    // This function returns `true` if the data have been queued, or `false` if a
    // shutdown request has been initiated.
    // This function is thread-safe.
    bool
    async_send(const void* data, size_t size);

    // Initiates normal closure of this stream.
    // The read stream is closed immediately. No further data will be received from
    // this socket, but the connection cannot be closed until all pending data are
    // delivered to the remote peer.
    // Note half-closed connections are not supported.
    void
    async_shutdown()
    noexcept;
  };

}  // namespace poseidon

#endif
