// This file is part of Poseidon.
// Copyleft 2020, LH_Mouse. All wrongs reserved.

#include "../precompiled.hpp"
#include "enums.hpp"
#include "../utils.hpp"

namespace poseidon {

IO_Result
get_io_result_from_errno(const char* func, int syserr)
  {
    switch(syserr) {
#if EAGAIN != EWOULDBLOCK
      case EAGAIN:
#endif
      case EWOULDBLOCK:
        return io_result_would_block;

      case EINTR:
      case 0:
        return io_result_partial_work;

      default:
        POSEIDON_THROW("i/O syscall error\n[`$1()` failed: $2]",
                       func, format_errno(syserr));
    }
  }

}  // namespace poseidon
