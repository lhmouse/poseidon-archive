// This file is part of Poseidon.
// Copyleft 2020, LH_Mouse. All wrongs reserved.

#include "../precompiled.hpp"
#include "zlib_stream_common.hpp"
#include "../utils.hpp"

namespace poseidon {
namespace details_zlib_stream_common {

void
zlib_Stream_Common::
do_throw_zlib_error(const char* func, int err)
  const
  {
    const char* msg = this->m_strm->msg;
    if(!msg)
      msg = "[no message]";

    POSEIDON_THROW("zlib error: $1\n[`$2()` returned $3]",
                   msg, func, err);
  }

void
zlib_Stream_Common::
do_reserve_output_buffer()
  {
    // Ensure there is enough space in the output buffer.
    size_t navail = this->m_obuf.reserve(64);
    this->m_strm->next_out = reinterpret_cast<uint8_t*>(this->m_obuf.mut_end());
    this->m_strm->avail_out = static_cast<uint32_t>(navail);
  }

void
zlib_Stream_Common::
do_update_output_buffer()
  noexcept
  {
    // Consume output bytes, if any.
    auto pbase = reinterpret_cast<const uint8_t*>(this->m_obuf.end());
    this->m_obuf.accept(static_cast<size_t>(this->m_strm->next_out - pbase));
  }

}  // namespace details_zlib_stream_common
}  // namespace poseidon
