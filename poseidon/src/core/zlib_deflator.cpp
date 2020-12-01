// This file is part of Poseidon.
// Copyleft 2020, LH_Mouse. All wrongs reserved.

#include "../precompiled.hpp"
#include "zlib_deflator.hpp"
#include "../util.hpp"

namespace poseidon {

zlib_Deflator::
zlib_Deflator(Format fmt, int level)
  : m_strm()
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

      default:
        POSEIDON_THROW("Invalid zlib deflator format: $1", fmt);
    }

    // Create a deflate stream.
    int res = ::deflateInit2(&(this->m_strm), level, Z_DEFLATED,
                             wbits, 9, Z_DEFAULT_STRATEGY);
    if(res != Z_OK)
      this->do_throw_zlib_error("deflateInit2", res);
  }

zlib_Deflator::
~zlib_Deflator()
  {
    ::deflateEnd(&(this->m_strm));
  }

void
zlib_Deflator::
do_throw_zlib_error(const char* func, int err)
  const
  {
    const char* msg = this->m_strm.msg;
    if(!msg)
      msg = "[no message]";

    POSEIDON_THROW("zlib error: $1\n[`$2()` returned $3]",
                   msg, func, err);
  }

void
zlib_Deflator::
do_reserve_output_buffer()
  {
    // Ensure there is some space in the output buffer.
    if((this->m_strm.next_out == reinterpret_cast<uint8_t*>(
               this->m_obuf.mut_end())) && (this->m_strm.avail_out > 64))
      return;

    uint32_t navail = static_cast<uint32_t>(this->m_obuf.reserve(256));
    this->m_strm.next_out = reinterpret_cast<uint8_t*>(
                                          this->m_obuf.mut_end());
    this->m_strm.avail_out = navail;
  }

::rocket::linear_buffer&
zlib_Deflator::
do_update_output_buffer()
  {
    return this->m_obuf.accept(static_cast<size_t>(this->m_strm.next_out -
             reinterpret_cast<uint8_t*>(this->m_obuf.mut_end())));
  }

void
zlib_Deflator::
reset()
  noexcept
  {
    int res = ::deflateReset(&(this->m_strm));
    ROCKET_ASSERT(res == Z_OK);

    this->m_obuf.clear();
  }

void
zlib_Deflator::
write(const void* data, size_t size)
  {
    auto bptr = static_cast<const uint8_t*>(data);
    auto eptr = bptr + size;
    for(;;) {
      // Put some bytes into the stream.
      uint32_t navail = static_cast<uint32_t>(
                            ::rocket::min(eptr - bptr, INT_MAX));
      this->m_strm.next_in = bptr;
      this->m_strm.avail_in = navail;
      if(navail == 0)
            break;

      while(this->m_strm.avail_in != 0) {
        // Extend the output buffer so we never get `Z_BUF_ERROR`.
        this->do_reserve_output_buffer();

        int res = ::deflate(&(this->m_strm), Z_NO_FLUSH);
        if(res != Z_OK)
          this->do_throw_zlib_error("deflate", res);
      }
      bptr += navail;
    }
  }

::rocket::linear_buffer&
zlib_Deflator::
synchronize()
  {
    // Put nothing, but force `Z_SYNC_FLUSH`. If there is nothing
    // to do, `deflate()` will return `Z_BUF_ERROR`.
    this->m_strm.next_in = nullptr;
    this->m_strm.avail_in = 0;

    this->do_reserve_output_buffer();

    int res = ::deflate(&(this->m_strm), Z_SYNC_FLUSH);
    if((res != Z_OK) && (res != Z_BUF_ERROR))
      this->do_throw_zlib_error("deflate", res);

    // The output buffer is now ready for consumption.
    return this->do_update_output_buffer();
  }

::rocket::linear_buffer&
zlib_Deflator::
finish()
  {
    // Put nothing, but force `Z_FINISH`. If there is nothing to
    // do, `deflate()` will return `Z_STREAM_END`.
    this->m_strm.next_in = nullptr;
    this->m_strm.avail_in = 0;

    this->do_reserve_output_buffer();

    int res = ::deflate(&(this->m_strm), Z_FINISH);
    if(res != Z_STREAM_END)
      this->do_throw_zlib_error("deflate", res);

    // The output buffer is now ready for consumption.
    return this->do_update_output_buffer();
  }

}  // namespace poseidon
