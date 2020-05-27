// This file is part of Poseidon.
// Copyleft 2020, LH_Mouse. All wrongs reserved.

#include "../precompiled.hpp"
#include "abstract_listen_socket.hpp"
#include "socket_address.hpp"
#include "../static/network_driver.hpp"
#include "../utilities.hpp"

namespace poseidon {
namespace {

IO_Result
do_translate_syscall_error(const char* func, int err)
  {
    if(err == EINTR)
      return io_result_not_eof;

    if(::rocket::is_any_of(err, { EAGAIN, EWOULDBLOCK }))
      return io_result_again;

    POSEIDON_THROW("failed to accept incoming connection\n"
                   "[`$1()` failed: $2]",
                   func, noadl::format_errno(err));
  }

}  // namespace

Abstract_Listen_Socket::
~Abstract_Listen_Socket()
  {
  }

void
Abstract_Listen_Socket::
do_set_common_options()
  {
    // Enable reusing addresses.
    static constexpr int yes[] = { -1 };
    int res = ::setsockopt(this->get_fd(), SOL_SOCKET, SO_REUSEADDR, yes, sizeof(yes));
    ROCKET_ASSERT(res == 0);
  }

IO_Result
Abstract_Listen_Socket::
do_on_async_poll_read(Si_Mutex::unique_lock& /*lock*/, void* /*hint*/, size_t /*size*/)
  try {
    // Try accepting a socket.
    Socket_Address::storage_type addrst;
    Socket_Address::size_type addrlen = sizeof(addrst);
    unique_FD fd(::accept4(this->get_fd(), addrst, &addrlen, SOCK_NONBLOCK));
    if(!fd)
      return do_translate_syscall_error("accept4", errno);

    // Create a new socket object.
    auto sock = this->do_on_async_accept(::std::move(fd));
    if(!sock)
      POSEIDON_THROW("null pointer returned from `do_on_async_accept()`\n"
                     "[acceptor socket class `$1`]",
                     typeid(*this).name());

    POSEIDON_LOG_INFO("Accepted incoming connection: local '$1', remote '$2'\n"
                      "[acceptor socket class `$3`; new socket class `$4`]",
                      sock->get_local_address(), Socket_Address(addrst, addrlen),
                      typeid(*this).name(), typeid(*sock).name());

    Network_Driver::insert(::std::move(sock));

    // Report success.
    return io_result_not_eof;
  }
  catch(exception& stdex) {
    // It is probably bad to let the exception propagate to network driver and kill
    // this server socket... so we catch and ignore this exception.
    POSEIDON_LOG_ERROR("Error accepting incoming connection: $2\n"
                       "[socket class `$1`]",
                       typeid(*this).name(), stdex.what());

    // Accept other connections. The error is considered non-fatal.
    return io_result_not_eof;
  }

size_t
Abstract_Listen_Socket::
do_write_queue_size(Si_Mutex::unique_lock& /*lock*/)
const
  {
    return 0;
  }

IO_Result
Abstract_Listen_Socket::
do_on_async_poll_write(Si_Mutex::unique_lock& /*lock*/, void* /*hint*/, size_t /*size*/)
  {
    return io_result_eof;
  }

void
Abstract_Listen_Socket::
do_on_async_poll_shutdown(int err)
  {
    POSEIDON_LOG_INFO("Listen socket closed: local '$1', $2",
                      this->get_local_address(), noadl::format_errno(err));
  }

void
Abstract_Listen_Socket::
do_listen(const Socket_Address& addr, int backlog)
  {
    // Bind onto `addr`.
    if(::bind(this->get_fd(), addr.data(), addr.size()) != 0)
      POSEIDON_THROW("failed to bind accept socket onto '$2'\n"
                     "[`bind()` failed: $1]",
                     noadl::format_errno(errno), addr);

    // Start listening.
    if(::listen(this->get_fd(), ::rocket::clamp(backlog, 1, SOMAXCONN)) != 0)
      POSEIDON_THROW("failed to set up listen socket on '$2'\n"
                     "[`listen()` failed: $1]",
                     noadl::format_errno(errno), this->get_local_address());

    POSEIDON_LOG_INFO("Listen socket opened: local '$1'",
                      this->get_local_address());
  }

}  // namespace poseidon
