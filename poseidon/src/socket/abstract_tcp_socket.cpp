// This file is part of Poseidon.
// Copyleft 2020, LH_Mouse. All wrongs reserved.

#include "../precompiled.hpp"
#include "abstract_tcp_socket.hpp"
#include "../util.hpp"
#include <netinet/tcp.h>

namespace poseidon {
namespace {

IO_Result
do_translate_syscall_error(const char* func, int err)
  {
    if(err == EINTR)
      return io_result_partial_work;

    if(::rocket::is_any_of(err, { EAGAIN, EWOULDBLOCK }))
      return io_result_would_block;

    POSEIDON_THROW("TCP socket error\n"
                   "[`$1()` failed: $2]",
                   func, format_errno(err));
  }

}  // namespace

Abstract_TCP_Socket::
~Abstract_TCP_Socket()
  {
  }

void
Abstract_TCP_Socket::
do_set_common_options()
  {
    // Disables Nagle algorithm.
    static constexpr int yes[] = { -1 };
    int res = ::setsockopt(this->get_fd(), IPPROTO_TCP, TCP_NODELAY,
                           yes, sizeof(yes));
    ROCKET_ASSERT(res == 0);
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
      return do_translate_syscall_error("read", errno);

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
      return do_translate_syscall_error("write", errno);

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
do_on_socket_establish()
  {
    POSEIDON_LOG_INFO("TCP connection established: local '$1', remote '$2'",
                      this->get_local_address(), this->get_remote_address());
  }

void
Abstract_TCP_Socket::
do_on_socket_close(int err)
  {
    POSEIDON_LOG_INFO("TCP connection closed: local '$1', $2",
                      this->get_local_address(), format_errno(err));
  }

}  // namespace poseidon
