// This file is part of Poseidon.
// Copyleft 2020, LH_Mouse. All wrongs reserved.

#include "../precompiled.hpp"
#include "abstract_accept_socket.hpp"
#include "socket_address.hpp"
#include "../utilities.hpp"

namespace poseidon {

Abstract_Accept_Socket::
~Abstract_Accept_Socket()
  {
  }

void
Abstract_Accept_Socket::
do_set_common_options()
  {
    // Enable reusing addresses.
    static constexpr int true_val[] = { -1 };
    int res = ::setsockopt(this->get_fd(), SOL_SOCKET, SO_REUSEADDR,
                           true_val, sizeof(true_val));
    ROCKET_ASSERT(res == 0);
  }

IO_Result
Abstract_Accept_Socket::
do_on_async_read(::rocket::mutex::unique_lock& /*lock*/, void* /*hint*/, size_t /*size*/)
  try {
    // Try accepting a socket.
    Socket_Address::storage_type addr;
    Socket_Address::size_type size = sizeof(addr);

    unique_posix_fd fd(::accept4(this->get_fd(), addr, &size, SOCK_NONBLOCK), ::close);
    if(!fd) {
      // Handle errors.
      int err = errno;
      if(err == EINTR)
        return io_result_intr;

      if(::rocket::is_any_of(err, { EAGAIN, EWOULDBLOCK }))
        return io_result_again;

      POSEIDON_THROW("error accepting incoming connection\n"
                     "[`accept4()` failed: $1]",
                     noadl::format_errno(err));
    }

    // Create a new socket object.
    auto sock = this->do_on_async_accept(::std::move(fd));
    if(!sock)
      POSEIDON_THROW("null pointer returned from `do_on_async_accept()`\n"
                     "[socket class `$1`]",
                     typeid(*this).name());

    POSEIDON_LOG_INFO("Accepted incoming connection: local '$1', remote '$2'",
                      sock->get_local_address(), sock->get_remote_address());
    // TODO register socket
    return io_result_not_eof;
  }
  catch(exception& stdex) {
    // It is probably bad to let the exception propagate to network driver and kill
    // this server socket... so we catch and ignore this exception.
    POSEIDON_LOG_ERROR("Error accepting incoming connection: $2\n"
                       "[socket class `$1`]",
                       typeid(*this).name(), stdex.what());

    // Accept other connections. The error is considered non-fatal.
    return io_result_intr;
  }

size_t
Abstract_Accept_Socket::
do_write_queue_size(::rocket::mutex::unique_lock& /*lock*/)
const
  {
    return 0;
  }

IO_Result
Abstract_Accept_Socket::
do_on_async_write(::rocket::mutex::unique_lock& /*lock*/, void* /*hint*/, size_t /*size*/)
  {
    return io_result_eof;
  }

void
Abstract_Accept_Socket::
do_on_async_shutdown(int err)
  {
    POSEIDON_LOG_INFO("Stopped listening on '$1': $2",
                      this->get_local_address(),
                      noadl::noadl::format_errno(err));
  }

void
Abstract_Accept_Socket::
bind_and_listen(const Socket_Address& addr, uint32_t backlog)
  {
    // Bind onto `addr`.
    if(::bind(this->get_fd(), addr.data(), addr.size()) != 0)
      POSEIDON_THROW("failed to bind socket onto '$2'\n",
                     "[`bind()` failed: $1]",
                     noadl::format_errno(errno), addr);

    // Start listening.
    static constexpr uint32_t backlog_min = 1;
    static constexpr uint32_t backlog_max = SOMAXCONN;

    if(::listen(this->get_fd(), static_cast<int>(::rocket::clamp(backlog,
                                                       backlog_min, backlog_max))) != 0)
      POSEIDON_THROW("failed to set up listen socket on '$2'\n",
                     "[`listen()` failed: $1]",
                     noadl::format_errno(errno), this->get_local_address());

     POSEIDON_LOG_INFO("Started listening on '$1'...", this->get_local_address());
  }

}  // namespace poseidon
