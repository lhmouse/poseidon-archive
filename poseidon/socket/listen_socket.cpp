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
Listen_Socket(const Socket_Address& saddr)
  : Abstract_Socket(SOCK_STREAM, IPPROTO_TCP)
  {
    // Use `SO_REUSEADDR`. Errors are ignored.
    int ival = 1;
    ::setsockopt(this->fd(), SOL_SOCKET, SO_REUSEADDR, &ival, sizeof(ival));

    // Bind this socket onto `addr`.
    ::sockaddr_in6 addr;
    addr.sin6_family = AF_INET6;
    addr.sin6_port = htobe16(saddr.port());
    addr.sin6_flowinfo = 0;
    addr.sin6_addr = saddr.addr();
    addr.sin6_scope_id = 0;

    if(::bind(this->fd(), (const ::sockaddr*) &addr, sizeof(addr)) != 0)
      POSEIDON_THROW((
          "Failed to bind TCP socket onto `$4`",
          "[`bind()` failed: $3]",
          "[TCP socket `$1` (class `$2`)]"),
          this, typeid(*this), format_errno(), saddr);

    if(::listen(this->fd(), SOMAXCONN) != 0)
      POSEIDON_THROW((
          "Failed to start listening on `$4`",
          "[`listen()` failed: $3]",
          "[TCP listen socket `$1` (class `$2`)]"),
          this, typeid(*this), format_errno(), saddr);

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
      unique_posix_fd fd(::accept4(this->fd(), nullptr, nullptr, SOCK_NONBLOCK));

      if(!fd) {
        if((errno == EAGAIN) || (errno == EWOULDBLOCK))
          break;

        POSEIDON_LOG_ERROR((
            "Error accepting TCP connection",
            "[`accept4()` failed: $3]",
            "[TCP listen socket `$1` (class `$2`)]"),
            this, typeid(*this), format_errno());

        // Errors are ignored.
        continue;
      }

      // Accept the client socket. If a null pointer is returned, the accepted
      // socket will be closed immediately.
      auto client = this->do_on_listen_new_client_opt(::std::move(fd));
      if(!client)
        continue;

      driver.insert(client);
    }
  }

void
Listen_Socket::
do_abstract_socket_on_oob_readable()
  {
  }

void
Listen_Socket::
do_abstract_socket_on_writable()
  {
  }

}  // namespace poseidon
