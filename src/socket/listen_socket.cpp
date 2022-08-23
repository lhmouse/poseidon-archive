// This file is part of Poseidon.
// Copyleft 2022, LH_Mouse. All wrongs reserved.

#include "../precompiled.ipp"
#include "listen_socket.hpp"
#include "../static/async_logger.hpp"
#include "../static/network_driver.hpp"
#include "../utils.hpp"
#include <sys/socket.h>

namespace poseidon {

Listen_Socket::
Listen_Socket(const Socket_Address& addr)
  :
    Abstract_Socket(addr.family(), SOCK_STREAM, IPPROTO_TCP)
  {
    // Use `SO_REUSEADDR`. Errors are ignored.
    int ival = 1;
    ::setsockopt(this->fd(), SOL_SOCKET, SO_REUSEADDR, &ival, sizeof(ival));

    if(::bind(this->fd(), addr.addr(), addr.ssize()) != 0)
      POSEIDON_THROW((
          "Failed to bind TCP socket onto `$4`",
          "[`bind()` failed: $3]",
          "[TCP socket `$1` (class `$2`)]"),
          this, typeid(*this), format_errno(), addr);

    if(::listen(this->fd(), SOMAXCONN) != 0)
      POSEIDON_THROW((
          "Failed to start listening on `$4`",
          "[`listen()` failed: $3]",
          "[TCP socket `$1` (class `$2`)]"),
          this, typeid(*this), format_errno(), addr);

    POSEIDON_LOG_INFO((
        "TCP server started listening on `$3`",
        "[TCP socket `$1` (class `$2`)]"),
        this, typeid(*this), this->get_local_address());
  }

Listen_Socket::
~Listen_Socket()
  {
  }

void
Listen_Socket::
do_abstract_socket_on_closed(int err)
  {
    POSEIDON_LOG_INFO((
        "TCP server stopped listening on `$3`: $4",
        "[TCP socket `$1` (class `$2`)]"),
        this, typeid(*this), this->get_local_address(), format_errno(err));
  }

void
Listen_Socket::
do_abstract_socket_on_readable()
  {
    recursive_mutex::unique_lock io_lock;
    auto& driver = this->do_abstract_socket_lock_driver(io_lock);

    // Try getting a connection.
    unique_posix_fd fd;
  try_io:
    fd.reset(::accept4(this->fd(), nullptr, nullptr, SOCK_NONBLOCK));
    if(!fd) {
      switch(errno) {
        case EINTR:
          goto try_io;

#if EWOULDBLOCK != EAGAIN
        case EAGAIN:
#endif
        case EWOULDBLOCK:
          return;
      }

      POSEIDON_THROW((
          "Error accepting TCP connection",
          "[`accept4()` failed: $3]",
          "[TCP socket `$1` (class `$2`)]"),
          this, typeid(*this), format_errno());
    }

    // Create the session object.
    auto client = this->do_on_listen_new_client_opt(::std::move(fd));
    if(!client)
      POSEIDON_THROW((
          "Null pointer returned from `do_on_new_client_opt()`",
          "[TCP socket `$1` (class `$2`)]"),
          this, typeid(*this));

    driver.insert(client);
  }

void
Listen_Socket::
do_abstract_socket_on_writable()
  {
  }

void
Listen_Socket::
do_abstract_socket_on_exception(exception& stdex)
  {
    POSEIDON_LOG_WARN((
        "Ignoring exception: $3",
        "[TCP socket `$1` (class `$2`)]"),
        this, typeid(*this), stdex);
  }

}  // namespace poseidon
