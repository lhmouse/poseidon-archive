// This file is part of Poseidon.
// Copyleft 2020, LH_Mouse. All wrongs reserved.

#include "../precompiled.hpp"
#include "zlib_inflator.hpp"
#include "../utils.hpp"

namespace poseidon {

zlib_Inflator::
zlib_Inflator(Format fmt)
  {
    // Get the `windowBits` argument.
    int wbits;
    switch(fmt) {
      case format_deflate:
        wbits = 15;
        break;

      case format_raw:
        wbits = -15;
        break;

      case format_gzip:
        wbits = 31;
        break;

      case format_auto:
        wbits = 47;
        break;

      default:
        POSEIDON_THROW("Invalid zlib inflator format: $1", fmt);
    }

    // Create an inflate stream.
    // Note this must be the last operation in each constructor.
    int res = ::inflateInit2(this->m_zlib, wbits);
    if(res != Z_OK)
      this->m_zlib.throw_zlib_error("inflateInit2", res);
  }

zlib_Inflator::
~zlib_Inflator()
  {
    ::inflateEnd(this->m_zlib);
  }

zlib_Inflator&
zlib_Inflator::
reset()
  noexcept
  {
    int res = ::inflateReset(this->m_zlib);
    ROCKET_ASSERT(res == Z_OK);

    this->m_zlib.strm.next_out = nullptr;
    this->m_zlib.strm.avail_out = 0;
    this->m_zlib.obuf.clear();
    return *this;
  }

zlib_Inflator&
zlib_Inflator::
write(const char* data, size_t size)
  {
    // Set up the read pointer.
    const auto eptr = reinterpret_cast<const uint8_t*>(data + size);
    this->m_zlib.strm.next_in = eptr - size;
    for(;;) {
      // The stupid zlib library uses a 32-bit integer for number of bytes.
      this->m_zlib.strm.avail_in = static_cast<uint32_t>(
                ::rocket::min(eptr - this->m_zlib.strm.next_in, INT32_MAX));
      if(this->m_zlib.strm.avail_in == 0)
        break;

      // Extend the output buffer so we never get `Z_BUF_ERROR`.
      this->m_zlib.reserve_output_buffer();
      int res = ::deflate(this->m_zlib, Z_NO_FLUSH);
      if(res != Z_OK)
        this->m_zlib.throw_zlib_error("deflate", res);

      this->m_zlib.update_output_buffer();
    }
    return *this;
  }

zlib_Inflator&
zlib_Inflator::
flush()
  {
    // Put nothing, but force `Z_SYNC_FLUSH`.
    this->m_zlib.strm.next_in = nullptr;
    this->m_zlib.strm.avail_in = 0;

    // If there is nothing to do, `inflate()` will return `Z_BUF_ERROR`.
    this->m_zlib.reserve_output_buffer();
    int res = ::inflate(this->m_zlib, Z_SYNC_FLUSH);
    if((res != Z_OK) && (res != Z_BUF_ERROR))
      this->m_zlib.throw_zlib_error("inflate", res);

    this->m_zlib.update_output_buffer();
    return *this;
  }

zlib_Inflator&
zlib_Inflator::
finish()
  {
    // Put nothing, but force `Z_FINISH`.
    this->m_zlib.strm.next_in = nullptr;
    this->m_zlib.strm.avail_in = 0;

    // If there is nothing to do, `inflate()` will return `Z_STREAM_END`.
    this->m_zlib.reserve_output_buffer();
    int res = ::inflate(this->m_zlib, Z_FINISH);
    if(res != Z_STREAM_END)
      this->m_zlib.throw_zlib_error("inflate", res);

    this->m_zlib.update_output_buffer();
    return *this;
  }

}  // namespace poseidon
