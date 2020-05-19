// This file is part of Poseidon.
// Copyleft 2020, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_NETWORK_ENUMS_HPP_
#define POSEIDON_NETWORK_ENUMS_HPP_

#include "../fwd.hpp"

namespace poseidon {

// This is the return type of I/O functions.
// Note that positive values denote number of bytes transferred.
// `io_result_not_eof` may be returned to indicate success of a
// non-stream operation, such as `accept()`, `recvfrom()`, or
// `shutdown(). I/O functions shall throw exceptions for errors
// that are not listed here.
enum IO_Result : ptrdiff_t
  {
    io_result_intr     = -2,  // EINTR
    io_result_again    = -1,  // EAGAIN or EWOULDBLOCK
    io_result_eof      =  0,
    io_result_not_eof  =  1,
  };

}  // namespace poseidon

#endif
