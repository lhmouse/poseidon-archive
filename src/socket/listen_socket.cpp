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
    // Create a new non-blocking socket.
    Abstract_Socket(addr.family(), SOCK_STREAM, IPPROTO_TCP)
  {
    // Use `SO_REUSEADDR`. Errors are ignored.
    int ival = 1;
    ::setsockopt(this->fd(), SOL_SOCKET, SO_REUSEADDR, &ival, sizeof(ival));

    if(::bind(this->fd(), addr.addr(), addr.ssize()) != 0)
      POSEIDON_THROW((
          "Failed to bind TCP socket onto `$4`",
          "[`bind()` failed: $3]",
          "[TCP listen socket `$1` (class `$2`)]"),
          this, typeid(*this), format_errno(), addr);

    if(::listen(this->fd(), SOMAXCONN) != 0)
      POSEIDON_THROW((
          "Failed to start listening on `$4`",
          "[`listen()` failed: $3]",
          "[TCP listen socket `$1` (class `$2`)]"),
          this, typeid(*this), format_errno(), addr);

    POSEIDON_LOG_INFO((
        "TCP server started listening on `$3`",
        "[TCP listen socket `$1` (class `$2`)]"),
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
        "[TCP listen socket `$1` (class `$2`)]"),
        this, typeid(*this), this->get_local_address(), format_errno(err));
  }

void
Listen_Socket::
do_abstract_socket_on_readable()
  {
    recursive_mutex::unique_lock io_lock;
    auto& driver = this->do_abstract_socket_lock_driver(io_lock);

    for(;;) {
      // Try getting a connection.
      Socket_Address addr;
      ::socklen_t addrlen = addr.capacity();
      unique_posix_fd fd(::accept4(this->fd(), addr.mut_addr(), &addrlen, SOCK_NONBLOCK));

      if(!fd) {
        if((errno == EAGAIN) || (errno == EWOULDBLOCK))
          break;

        POSEIDON_THROW((
            "Error accepting TCP connection",
            "[`accept4()` failed: $3]",
            "[TCP listen socket `$1` (class `$2`)]"),
            this, typeid(*this), format_errno());
      }

      // Accept the client socket. If a null pointer is returned, the accepted
      // socket will be closed immediately.
      addr.set_size(addrlen);
      auto client = this->do_on_listen_new_client_opt(::std::move(fd));
      if(!client)
        continue;

      POSEIDON_LOG_INFO((
          "Accepted new TCP connection from `$5`",
          "[TCP client socket `$3` (class `$4`)]",
          "[TCP listen socket `$1` (class `$2`)]"),
          this, typeid(*this), client, typeid(*client), addr);

      driver.insert(client);
    }
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
        "[TCP listen socket `$1` (class `$2`)]"),
        this, typeid(*this), stdex);
  }

}  // namespace poseidon
