// This file is part of Poseidon.
// Copyleft 2022 - 2023, LH_Mouse. All wrongs reserved.

#include "../precompiled.ipp"
#include "abstract_socket.hpp"
#include "../utils.hpp"
#include <sys/socket.h>
#include <fcntl.h>

namespace poseidon {

Abstract_Socket::
Abstract_Socket(unique_posix_fd&& fd)
  {
    // Take ownership the socket handle.
    this->m_fd = ::std::move(fd);
    if(!this->m_fd)
      POSEIDON_THROW(("Null socket handle not valid"));

    // Get the local address and address family.
    ::sockaddr_in6 addr;
    ::socklen_t addrlen = sizeof(addr);
    if(::getsockname(this->fd(), (::sockaddr*) &addr, &addrlen) != 0)
      POSEIDON_THROW((
          "Could not get socket local address",
          "[`getsockname()` failed: $1]"),
          format_errno());

    if((addr.sin6_family != AF_INET6) || (addrlen != sizeof(addr)))
      POSEIDON_THROW((
          "Addresss family unimplemented: family `$1`, addrlen `$2`"),
          addr.sin6_family, addrlen);

    this->m_sockname.set_addr(addr.sin6_addr);
    this->m_sockname.set_port(be16toh(addr.sin6_port));
    this->m_sockname_ready.store(true);

    // Turn on non-blocking mode if it hasn't been enabled.
    int fl_old = ::fcntl(this->fd(), F_GETFL);
    if(fl_old == -1)
      POSEIDON_THROW((
          "Could not get socket flags",
          "[`fcntl()` failed: $1]"),
          format_errno());

    int fl_new = fl_old | O_NONBLOCK;
    if(fl_new != fl_old)
      ::fcntl(this->fd(), F_SETFL, fl_new);

    this->m_state.store(socket_state_established);
  }

Abstract_Socket::
Abstract_Socket(int type, int protocol)
  {
    // Create a non-blocking socket.
    this->m_fd.reset(::socket(AF_INET6, type | SOCK_NONBLOCK, protocol));
    if(!this->m_fd)
      POSEIDON_THROW((
          "Could not create IPv6 socket: type `$2`, protocol `$3`",
          "[`socket()` failed: $1]"),
          format_errno(), type, protocol);

    this->m_state.store(socket_state_connecting);
  }

Abstract_Socket::
~Abstract_Socket()
  {
  }

const Socket_Address&
Abstract_Socket::
local_address() const noexcept
  {
    if(this->m_sockname_ready.load())
      return this->m_sockname;

    // Try getting the address now.
    static plain_mutex s_mutex;
    plain_mutex::unique_lock lock(s_mutex);

    if(this->m_sockname_ready.load())
      return this->m_sockname;

    ::sockaddr_in6 addr;
    ::socklen_t addrlen = sizeof(addr);
    if(::getsockname(this->fd(), (::sockaddr*) &addr, &addrlen) != 0)
      return ipv6_unspecified;

    ROCKET_ASSERT(addr.sin6_family == AF_INET6);
    ROCKET_ASSERT(addrlen == sizeof(addr));

    // Cache the address.
    this->m_sockname.set_addr(addr.sin6_addr);
    this->m_sockname.set_port(be16toh(addr.sin6_port));
    this->m_sockname_ready.store(true);
    return this->m_sockname;
  }

bool
Abstract_Socket::
quick_shut_down() noexcept
  {
    this->m_state.store(socket_state_closed);

    // Enable linger to request that any pending data be discarded.
    ::linger lng;
    lng.l_onoff = 1;
    lng.l_linger = 0;
    ::setsockopt(this->fd(), SOL_SOCKET, SO_LINGER, &lng, sizeof(lng));

    return ::shutdown(this->fd(), SHUT_RDWR) == 0;
  }

}  // namespace poseidon
