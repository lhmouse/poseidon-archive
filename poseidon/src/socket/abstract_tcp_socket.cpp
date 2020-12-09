// This file is part of Poseidon.
// Copyleft 2020, LH_Mouse. All wrongs reserved.

#include "../precompiled.hpp"
#include "abstract_tcp_socket.hpp"
#include "../utils.hpp"

namespace poseidon {

Abstract_TCP_Socket::
Abstract_TCP_Socket(unique_FD&& fd)
  : Abstract_Stream_Socket(::std::move(fd))
  {
  }

Abstract_TCP_Socket::
Abstract_TCP_Socket(::sa_family_t family)
  : Abstract_Stream_Socket(family)
  {
  }

Abstract_TCP_Socket::
~Abstract_TCP_Socket()
  {
  }

void
Abstract_TCP_Socket::
do_stream_preconnect_unlocked()
  {
  }

IO_Result
Abstract_TCP_Socket::
do_stream_read_unlocked(char*& data, size_t size)
  {
    ::ssize_t nread = ::read(this->get_fd(), data, size);
    if(nread < 0)
      return get_io_result_from_errno("read", errno);

    if(nread == 0)
      return io_result_end_of_stream;

    data += nread;
    return io_result_partial_work;
  }

IO_Result
Abstract_TCP_Socket::
do_stream_write_unlocked(const char*& data, size_t size)
  {
    ::ssize_t nwritten = ::write(this->get_fd(), data, size);
    if(nwritten < 0)
      return get_io_result_from_errno("write", errno);

    data += nwritten;
    return io_result_partial_work;
  }

IO_Result
Abstract_TCP_Socket::
do_stream_preclose_unlocked()
  {
    return io_result_end_of_stream;
  }

void
Abstract_TCP_Socket::
do_socket_on_establish()
  {
    POSEIDON_LOG_INFO("TCP connection established: local '$1', remote '$2'",
                      this->get_local_address(), this->get_remote_address());
  }

void
Abstract_TCP_Socket::
do_socket_on_close(int err)
  {
    POSEIDON_LOG_INFO("TCP connection closed: local '$1', $2",
                      this->get_local_address(), format_errno(err));
  }

}  // namespace poseidon
