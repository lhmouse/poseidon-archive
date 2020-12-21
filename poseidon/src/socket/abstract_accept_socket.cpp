// This file is part of Poseidon.
// Copyleft 2020, LH_Mouse. All wrongs reserved.

#include "../precompiled.hpp"
#include "abstract_accept_socket.hpp"
#include "../static/network_driver.hpp"
#include "../utils.hpp"

namespace poseidon {

Abstract_Accept_Socket::
Abstract_Accept_Socket(::sa_family_t family)
  : Abstract_Socket(family, SOCK_STREAM, IPPROTO_TCP)
  {
  }

Abstract_Accept_Socket::
~Abstract_Accept_Socket()
  {
  }

IO_Result
Abstract_Accept_Socket::
do_socket_on_poll_read(simple_mutex::unique_lock& lock, char* /*hint*/, size_t /*size*/)
  {
    lock.lock(this->m_io_mutex);

    try {
      // Try accepting a socket.
      Socket_Address::storage addrst;
      ::socklen_t addrlen = sizeof(addrst);
      unique_FD fd(::accept4(this->get_fd(), addrst, &addrlen, SOCK_NONBLOCK));
      if(!fd)
        return get_io_result_from_errno("accept4", errno);

      // Create a new socket object.
      lock.unlock();
      Socket_Address addr(addrst, addrlen);
      auto sock = this->do_socket_on_accept(::std::move(fd), addr);
      if(!sock)
        POSEIDON_THROW("Null pointer returned from `do_socket_on_accept()`\n"
                       "[listen socket class `$1`]",
                       typeid(*this));

      // Register the socket.
      POSEIDON_LOG_INFO("Accepted incoming connection from '$1'\n"
                        "[server socket class `$2` listening on '$3']\n"
                        "[accepted socket class `$4`]",
                        addr, typeid(*this), this->get_local_address(), typeid(*sock));

      this->do_socket_on_register(Network_Driver::insert(::std::move(sock)));
    }
    catch(exception& stdex) {
      // It is probably bad to let the exception propagate to network driver and kill
      // this server socket... so we catch and ignore this exception.
      POSEIDON_LOG_ERROR("Socket accept error: $1\n"
                         "[socket class `$2`]",
                         stdex, typeid(*this));
    }

    lock.lock(this->m_io_mutex);
    return io_result_partial_work;
  }

size_t
Abstract_Accept_Socket::
do_write_queue_size(simple_mutex::unique_lock& /*lock*/)
  const
  {
    return 0;
  }

IO_Result
Abstract_Accept_Socket::
do_socket_on_poll_write(simple_mutex::unique_lock& lock, char* /*hint*/, size_t /*size*/)
  {
    lock.lock(this->m_io_mutex);
    return io_result_end_of_stream;
  }

void
Abstract_Accept_Socket::
do_socket_on_poll_close(int err)
  {
    simple_mutex::unique_lock lock(this->m_io_mutex);
    this->m_cstate = connection_state_closed;
    lock.unlock();

    this->do_socket_on_close(err);
  }

void
Abstract_Accept_Socket::
do_socket_listen(const Socket_Address& addr, int backlog)
  {
    simple_mutex::unique_lock lock(this->m_io_mutex);
    if(this->m_cstate != connection_state_empty)
      POSEIDON_THROW("Socket state error (fresh socket expected)");

    static constexpr int yes[] = { -1 };
    int res = ::setsockopt(this->get_fd(), SOL_SOCKET, SO_REUSEADDR, yes, sizeof(yes));
    ROCKET_ASSERT(res == 0);

    if(::bind(this->get_fd(), addr.data(), addr.ssize()) != 0)
      POSEIDON_THROW("Failed to bind accept socket onto '$2'\n"
                     "[`bind()` failed: $1]",
                     format_errno(errno), addr);

    if(::listen(this->get_fd(), ::rocket::clamp(backlog, 1, SOMAXCONN)) != 0)
      POSEIDON_THROW("Failed to set up listen socket on '$2'\n"
                     "[`listen()` failed: $1]",
                     format_errno(errno), this->get_local_address());

    // Mark this socket listening.
    this->m_cstate = connection_state_established;

    POSEIDON_LOG_INFO("Accept socket listening: local '$1'",
                      this->get_local_address());
  }

void
Abstract_Accept_Socket::
do_socket_on_close(int err)
  {
    POSEIDON_LOG_INFO("Accept socket closed: local '$1', reason: $2",
                      this->get_local_address(), format_errno(err));
  }

bool
Abstract_Accept_Socket::
close()
  noexcept
  {
    simple_mutex::unique_lock lock(this->m_io_mutex);
    if(this->m_cstate > connection_state_established)
      return false;

    // Initiate asynchronous shutdown.
    ::shutdown(this->get_fd(), SHUT_RDWR);
    this->m_cstate = connection_state_closed;
    POSEIDON_LOG_TRACE("Marked accept socket as CLOSED (not open): $1", this);
    lock.unlock();

    // Notify the driver about availability of outgoing data.
    Network_Driver::notify_writable_internal(this);
    return true;
  }

}  // namespace poseidon
