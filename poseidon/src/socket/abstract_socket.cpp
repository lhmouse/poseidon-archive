// This file is part of Poseidon.
// Copyleft 2020, LH_Mouse. All wrongs reserved.

#include "../precompiled.hpp"
#include "abstract_socket.hpp"
#include "../utils.hpp"
#include <sys/socket.h>

namespace poseidon {

Abstract_Socket::
Abstract_Socket(unique_FD&& fd)
  {
    // Adopt the socket and enable non-blocking mode.
    this->m_fd = ::std::move(fd);

    int flags = ::fcntl(this->m_fd, F_GETFL);
    if(!(flags & O_NONBLOCK))
      ::fcntl(this->m_fd, F_SETFL, flags | O_NONBLOCK);
  }

Abstract_Socket::
Abstract_Socket(::sa_family_t family, int type, int protocol)
  {
    // Create a non-blocking socket.
    this->m_fd.reset(::socket(family, type | SOCK_NONBLOCK, protocol));
    if(!this->m_fd)
      POSEIDON_THROW(
        "Could not create socket (family `$2`, type `$3`, protocol `$4`)\n"
        "[`socket()` failed: $1]",
        format_errno(errno), family, type, protocol);
  }

Abstract_Socket::
~Abstract_Socket()
  {
  }

void
Abstract_Socket::
kill() noexcept
  {
    // Enable linger to discard all pending data.
    // Failure to set this option is ignored.
    ::linger lng;
    lng.l_onoff = 1;
    lng.l_linger = 0;
    ::setsockopt(this->get_fd(), SOL_SOCKET, SO_LINGER, &lng, sizeof(lng));

    // Shut down both directions.
    ::shutdown(this->get_fd(), SHUT_RDWR);
  }

const Socket_Address&
Abstract_Socket::
get_local_address() const
  {
    this->m_local_addr_once.call(
      [this] {
        // Try getting the local address.
        Socket_Address::storage addrst;
        ::socklen_t addrlen = sizeof(addrst);
        if(::getsockname(this->get_fd(), addrst, &addrlen) != 0)
          POSEIDON_THROW("could not get local socket address\n"
                         "[`getsockname()` failed: $1]",
                         format_errno(errno));

        // The result is cached once it becomes available.
        this->m_local_addr.assign(addrst, addrlen);
      });
    return this->m_local_addr;
  }

}  // namespace poseidon
