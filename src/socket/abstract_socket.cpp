// This file is part of Poseidon.
// Copyleft 2022, LH_Mouse. All wrongs reserved.

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
Abstract_Socket(int family, int type, int protocol)
  {
    // Create a non-blocking socket.
    this->m_fd.reset(::socket(family, type | SOCK_NONBLOCK, protocol));
    if(!this->m_fd)
      POSEIDON_THROW((
          "Could not create socket: family `$2`, type `$3`, protocol `$4`",
          "[`fcntl()` failed: $1]"),
          format_errno(), family, type, protocol);

    this->m_state.store(socket_state_connecting);
  }

Abstract_Socket::
~Abstract_Socket()
  {
  }

const cow_string&
Abstract_Socket::
get_local_address() const
  {
    if(!this->m_sockname_ready.load()) {
      plain_mutex::unique_lock lock(this->m_sockname_mutex);

      // Try getting the address.
      Socket_Address addr;
      ::socklen_t addrlen = (::socklen_t) addr.capacity();
      if(::getsockname(this->fd(), addr.mut_addr(), &addrlen) != 0)
        return format(this->m_sockname, "(unknown address: $1)", format_errno());

      // Cache the result.
      addr.set_size(addrlen);
      this->m_sockname = addr.format();
      this->m_sockname_ready.store(true);
    }
    return this->m_sockname;
  }

bool
Abstract_Socket::
quick_shut_down() noexcept
  {
    // Enable linger to request that any pending data be discarded.
    ::linger lng;
    lng.l_onoff = 1;
    lng.l_linger = 0;
    ::setsockopt(this->fd(), SOL_SOCKET, SO_LINGER, &lng, sizeof(lng));

    int status = ::shutdown(this->fd(), SHUT_RDWR);
    this->m_state.store(socket_state_closed);
    return status == 0;
  }

}  // namespace poseidon
