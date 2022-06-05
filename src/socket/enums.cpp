// This file is part of Poseidon.
// Copyleft 2020, LH_Mouse. All wrongs reserved.

#include "../precompiled.ipp"
#include "enums.hpp"
#include "../utils.hpp"

namespace poseidon {

IO_Result
get_io_result_from_errno(const char* func, int err)
  {
    switch(err) {
#if EAGAIN != EWOULDBLOCK
      case EAGAIN:
#endif
      case EWOULDBLOCK:
        return io_result_would_block;

      case EINTR:
      case 0:
        return io_result_partial_work;

      case EPIPE:
        return io_result_end_of_stream;

      default:
        POSEIDON_THROW("`$1()` failed: $2", func, format_errno(err));
    }
  }

}  // namespace poseidon
