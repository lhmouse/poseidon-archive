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
    using destructor  = int (::z_stream* strm);

    ::z_stream m_strm[1] = { };
    ::rocket::linear_buffer m_obuf;

    constexpr
    zlib_Stream_Common()
      noexcept
      = default;

    ASTERIA_NONCOPYABLE_DESTRUCTOR(zlib_Stream_Common)
      {
        // Abuse the opaque pointer for the destructor.
        auto dtor = reinterpret_cast<destructor*>(this->m_strm->opaque);
        if(dtor)
          (*dtor)(this->m_strm);
      }

    void
    do_set_destructor(destructor* dtor)
      noexcept
      { this->m_strm->opaque = reinterpret_cast<void*>(dtor);  }

    [[noreturn]]
    void
    do_throw_zlib_error(const char* func, int err)
      const;

    void
    do_reserve_output_buffer();

    void
    do_update_output_buffer()
      noexcept;
  };

}  // namespace details_zlib_stream_common
}  // namespace poseidon

#endif
