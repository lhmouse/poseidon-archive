// This file is part of Poseidon.
// Copyleft 2020, LH_Mouse. All wrongs reserved.

#include "../precompiled.hpp"
#include "zlib_stream_common.hpp"
#include "../utils.hpp"

namespace poseidon {
namespace details_zlib_stream_common {

void
zlib_Stream_Common::
throw_zlib_error(const char* func, int err)
  const
  {
    const char* msg = this->strm.msg;
    if(!msg)
      msg = "[no message]";

    POSEIDON_THROW("zlib error: $1\n[`$2()` returned $3]",
                   msg, func, err);
  }

void
zlib_Stream_Common::
reserve_output_buffer()
  {
    // Ensure there is enough space in the output buffer.
    size_t navail = this->obuf.reserve(64);
    this->strm.next_out = reinterpret_cast<uint8_t*>(this->obuf.mut_end());
    this->strm.avail_out = static_cast<uint32_t>(navail);
  }

void
zlib_Stream_Common::
update_output_buffer()
  noexcept
  {
    // Consume output bytes, if any.
    auto pbase = reinterpret_cast<const uint8_t*>(this->obuf.end());
    this->obuf.accept(static_cast<size_t>(this->strm.next_out - pbase));
  }

}  // namespace details_zlib_stream_common
}  // namespace poseidon
