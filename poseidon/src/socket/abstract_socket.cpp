// This file is part of Poseidon.
// Copyleft 2020, LH_Mouse. All wrongs reserved.

#include "../precompiled.hpp"
#include "abstract_socket.hpp"
#include "../util.hpp"
#include <sys/socket.h>

namespace poseidon {

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
terminate()
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

const Socket_Address&
Abstract_Socket::
get_local_address()
  const
  {
    this->m_local_addr_once.call(
      [this] {
        // Try getting the local address.
        Socket_Address::storage addrst;
        ::socklen_t addrlen = sizeof(addrst);
        if(::getsockname(this->get_fd(), addrst, &addrlen) != 0)
          POSEIDON_THROW("Could not get local socket address\n"
                         "[`getsockname()` failed: $1]",
                         format_errno(errno));

        // Cache the result.
        this->m_local_addr.assign(addrst, addrlen);
      });

    return this->m_local_addr;
  }

}  // namespace poseidon
