// This file is part of Poseidon.
// Copyleft 2022, LH_Mouse. All wrongs reserved.

#include "../precompiled.ipp"
#include "abstract_socket.hpp"
#include "../utils.hpp"
#include <fcntl.h>
#include <sys/socket.h>

namespace poseidon {

Abstract_Socket::
Abstract_Socket(unique_posix_fd&& fd)
  {
   // Take ownership the socket handle.
   this->m_fd = ::std::move(fd);
   if(!this->m_fd)
     POSEIDON_THROW(("Null socket handle not valid"));

   // Turn on non-blocking mode if it hasn't been enabled.
   int fl_old = ::fcntl(this->m_fd, F_GETFL);
   if(fl_old == -1)
     POSEIDON_THROW((
         "Could not get socket flags",
         "[`fcntl()` failed: $1]"),
         format_errno());

   int fl_new = fl_old | O_NONBLOCK;
   if(fl_new != fl_old)
     ::fcntl(this->fd(), F_SETFL, fl_new);
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
  }

Abstract_Socket::
~Abstract_Socket()
  {
  }

bool
Abstract_Socket::
close() noexcept
  {
    // Perform normal shutdown.
    int err = ::shutdown(this->fd(), SHUT_RDWR);
    this->m_state.store(socket_state_closed);
    return err == 0;
  }

bool
Abstract_Socket::
abort() noexcept
  {
    // Discard pending data. Errors are ignored.
    ::linger lng;
    lng.l_onoff = 1;
    lng.l_linger = 0;
    ::setsockopt(this->fd(), SOL_SOCKET, SO_LINGER, &lng, sizeof(lng));

    // Perform normal shutdown.
    return this->close();
  }

const Socket_Address&
Abstract_Socket::
get_local_address() const
  {
    // Get the socket name and cache it.
    this->m_sockname_once.call(
      [this] {
        ::socklen_t addrlen = (::socklen_t) this->m_sockname.capacity();
        if(::getsockname(this->fd(), this->m_sockname.mut_addr(), &addrlen) != 0)
          POSEIDON_THROW((
              "Could not get local address of socket",
              "[`getsockname()` failed: $1]"),
              format_errno());

        // Accept the address.
        this->m_sockname.set_size(addrlen);
      });
    return this->m_sockname;
  }

}  // namespace poseidon
