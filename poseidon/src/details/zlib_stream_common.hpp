// This file is part of Poseidon.
// Copyleft 2020, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_DETAILS_ZLIB_STREAM_COMMON_HPP_
#define POSEIDON_DETAILS_ZLIB_STREAM_COMMON_HPP_

#include "../fwd.hpp"
#define ZLIB_CONST 1
#include <zlib.h>

namespace poseidon {
namespace details_zlib_stream_common {

struct zlib_Stream_Common
  {
    ::z_stream strm = { };
    ::rocket::linear_buffer obuf;

    constexpr
    zlib_Stream_Common()
      noexcept
      = default;

    [[noreturn]]
    void
    throw_zlib_error(const char* func, int err)
      const;

    void
    reserve_output_buffer();

    void
    update_output_buffer()
      noexcept;

    constexpr operator
    const ::z_stream*()
      const noexcept
      { return &(this->strm);  }

    constexpr operator
    ::z_stream*()
      noexcept
      { return &(this->strm);  }
  };

}  // namespace details_zlib_stream_common
}  // namespace poseidon

#endif
