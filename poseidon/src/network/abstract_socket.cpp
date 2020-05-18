// This file is part of Poseidon.
// Copyleft 2020, LH_Mouse. All wrongs reserved.

#include "../precompiled.hpp"
#include "abstract_socket.hpp"
#include "socket_address.hpp"
#include "../utilities.hpp"

namespace poseidon {

Abstract_Socket::
~Abstract_Socket()
  {
  }

void
Abstract_Socket::
terminate()
noexcept
  {
    // Enable linger to discard all pending data.
    // Failure to set this option is ignored.
    ::linger lng;
    lng.l_onoff = 1;
    lng.l_linger = 0;
    ::setsockopt(this->m_fd, SOL_SOCKET, SO_LINGER, &lng, sizeof(lng));

    // Shut down both directions.
    ::shutdown(this->m_fd, SHUT_RDWR);
  }

Socket_Address
Abstract_Socket::
get_local_address()
const
  {
    Socket_Address::storage_type addr;
    Socket_Address::size_type size = sizeof(addr);

    if(::getsockname(this->m_fd, addr, &size) != 0)
      POSEIDON_THROW("could not get local socket address\n"
                     "[`getsockname()` failed: $1]'",
                     format_errno(errno));

    return Socket_Address(addr, size);
  }

Socket_Address
Abstract_Socket::
get_remote_address()
const
  {
    Socket_Address::storage_type addr;
    Socket_Address::size_type size = sizeof(addr);

    if(::getpeername(this->m_fd, addr, &size) != 0)
      POSEIDON_THROW("could not get remote socket address\n"
                     "[`getpeername()` failed: $1]'",
                     format_errno(errno));

    return Socket_Address(addr, size);
  }

}  // namespace poseidon
