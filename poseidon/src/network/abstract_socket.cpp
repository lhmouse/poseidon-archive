// This file is part of Poseidon.
// Copyleft 2020, LH_Mouse. All wrongs reserved.

#include "../precompiled.hpp"
#include "abstract_socket.hpp"
#include "socket_address.hpp"
#include "../utilities.hpp"
#include <sys/socket.h>

namespace poseidon {

unique_posix_fd
Abstract_Socket::
do_create_socket(int family, int type, int protocol)
  {
    unique_posix_fd fd(::socket(family, type, protocol), ::close);
    if(!fd)
      POSEIDON_THROW("could not create socket (family `$2`, type `$3`, protocol `$4`)\n"
                     "[`getsockname()` failed: $1]",
                     noadl::format_errno(errno), family, type, protocol);
    return fd;
  }

Abstract_Socket::
~Abstract_Socket()
  {
  }

void
Abstract_Socket::
do_set_common_options()
  {
    // Enable non-blocking mode.
    int fl_old = ::fcntl(this->get_fd(), F_GETFL);
    int fl_new = fl_old | O_NONBLOCK;
    if(fl_old != fl_new) {
      int res = ::fcntl(this->get_fd(), F_SETFL, fl_new);
      ROCKET_ASSERT(res == 0);
    }
  }

void
Abstract_Socket::
abort()
noexcept
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

Socket_Address
Abstract_Socket::
get_local_address()
const
  {
    // Try getting the local address.
    Socket_Address::storage_type addrst;
    Socket_Address::size_type addrlen = sizeof(addrst);
    if(::getsockname(this->get_fd(), addrst, &addrlen) != 0)
      POSEIDON_THROW("could not get local socket address\n"
                     "[`getsockname()` failed: $1]",
                     noadl::format_errno(errno));
    return { addrst, addrlen };
  }

}  // namespace poseidon
