// This file is part of Poseidon.
// Copyleft 2020, LH_Mouse. All wrongs reserved.

#include "../precompiled.hpp"
#include "abstract_tcp_socket.hpp"
#include "socket_address.hpp"
#include "../utilities.hpp"
#include <netinet/tcp.h>

namespace poseidon {
namespace {

IO_Result
do_translate_syscall_error(const char* func, int err)
  {
    if(err == EINTR)
      return io_result_intr;

    if(::rocket::is_any_of(err, { EAGAIN, EWOULDBLOCK }))
      return io_result_again;

    POSEIDON_THROW("TCP socket error\n"
                   "[`$1()` failed: $2]",
                   func, noadl::format_errno(err));
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
    static constexpr int true_val[] = { -1 };
    int res = ::setsockopt(this->get_fd(), IPPROTO_TCP, TCP_NODELAY,
                           true_val, sizeof(true_val));
    ROCKET_ASSERT(res == 0);
  }

void
Abstract_TCP_Socket::
do_stream_preconnect_nolock()
  {
  }

IO_Result
Abstract_TCP_Socket::
do_stream_read_nolock(void* data, size_t size)
  {
    ::ssize_t nread = ::read(this->get_fd(), data, size);
    if(nread > 0)
      return static_cast<IO_Result>(nread);

    if(nread == 0)
      return io_result_eof;

    return do_translate_syscall_error("read", errno);
  }

IO_Result
Abstract_TCP_Socket::
do_stream_write_nolock(const void* data, size_t size)
  {
    ::ssize_t nread = ::write(this->get_fd(), data, size);
    if(nread > 0)
      return static_cast<IO_Result>(nread);

    if(nread == 0)
      return io_result_not_eof;

    return do_translate_syscall_error("write", errno);
  }

IO_Result
Abstract_TCP_Socket::
do_stream_preshutdown_nolock()
  {
    return io_result_eof;
  }

void
Abstract_TCP_Socket::
do_on_async_establish()
  {
    POSEIDON_LOG_INFO("TCP connection established: local '$1', remote '$2'",
                      this->get_local_address(), this->get_remote_address());
  }

void
Abstract_TCP_Socket::
do_on_async_shutdown(int err)
  {
    POSEIDON_LOG_INFO("TCP connection closed: local '$1', reason: $2",
                      this->get_local_address(), noadl::format_errno(err));
  }

}  // namespace poseidon
