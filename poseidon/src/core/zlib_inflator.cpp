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

    // Construct the stream now.
    this->do_construct(0, wbits, ::inflateEnd);
  }

zlib_Inflator::
~zlib_Inflator()
  {
  }

void
zlib_Inflator::
do_zlib_construct(::z_stream* strm, int /*level*/, int wbits)
  {
    int res = ::inflateInit2(strm, wbits);
    if(res != Z_OK)
      this->do_zlib_throw_error("inflateInit2", res);
  }

void
zlib_Inflator::
do_zlib_reset(::z_stream* strm)
  {
    int res = ::inflateReset(strm);
    if(res != Z_OK)
      this->do_zlib_throw_error("inflateReset", res);
  }

void
zlib_Inflator::
do_zlib_write_partial(int& res, ::z_stream* strm, int flush)
  {
    res = ::inflate(strm, flush);
    if(::rocket::is_none_of(res, { Z_OK, Z_STREAM_END, Z_BUF_ERROR }))
      this->do_zlib_throw_error("inflate", res);
  }

}  // namespace poseidon
