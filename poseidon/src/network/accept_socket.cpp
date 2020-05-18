// This file is part of Poseidon.
// Copyleft 2020, LH_Mouse. All wrongs reserved.

#include "../precompiled.hpp"
#include "accept_socket.hpp"
#include "socket_address.hpp"
#include "../utilities.hpp"

namespace poseidon {

Accept_Socket::
~Accept_Socket()
  {
  }

void
Accept_Socket::
do_set_common_options()
  {
    // Enable reusing addresses.
    int val = 1;
    ::setsockopt(this->get_fd(), SOL_SOCKET, SO_REUSEADDR, &val, sizeof(val));
  }

/*IO_Result
do_on_async_read(::rocket::mutex::unique_lock& lock, void* hint, size_t size)
override;

// Does nothing.
// This function always returns zero.
// `lock` is ignored.
size_t
do_write_queue_size(::rocket::mutex::unique_lock& lock)
const noexcept override;

// Does nothing.
// This function always returns `io_result_end_of_stream`.
// `lock` is ignored.
IO_Result
do_on_async_write(::rocket::mutex::unique_lock& lock, void* hint, size_t size)
override;

// Prints a line of text but does nothing otherwise.
void
do_on_async_shutdown(int err)
override;

blic:
// Binds this socket to the specified address and starts listening.
// `backlog` is clamped between `1` and `SOMAXCONN`. Out-of-bound values
// are truncated silently.
void
bind_and_listen(const Socket_Address& addr, uint32_t backlog = UINT32_MAX);
*/
}  // namespace poseidon
