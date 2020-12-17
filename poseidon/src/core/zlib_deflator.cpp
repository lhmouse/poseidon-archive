// This file is part of Poseidon.
// Copyleft 2020, LH_Mouse. All wrongs reserved.

#include "../precompiled.hpp"
#include "zlib_deflator.hpp"
#include "../static/main_config.hpp"
#include "../core/config_file.hpp"
#include "../utils.hpp"

namespace poseidon {

zlib_Deflator::
zlib_Deflator(Format fmt, int level)
  {
    // If `level` equals `Z_DEFAULT_COMPRESSION` and a default level is
    // specified in 'main.conf', use that default level.
    int rlevel = level;
    if(rlevel == Z_DEFAULT_COMPRESSION) {
      const auto file = Main_Config::copy();

      auto qint = file.get_int64_opt({"general","default_compression_level"});
      if(qint)
        rlevel = static_cast<int>(::rocket::clamp(*qint, 0, 9));
    }

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

    // Construct the stream now.
    this->do_construct(rlevel, wbits, ::deflateEnd);
  }

zlib_Deflator::
~zlib_Deflator()
  {
  }

void
zlib_Deflator::
do_zlib_construct(::z_stream* strm, int level, int wbits)
  {
    int res = ::deflateInit2(strm, level, Z_DEFLATED, wbits, 9, 0);
    if(res != Z_OK)
      this->do_zlib_throw_error("deflateInit2", res);
  }

void
zlib_Deflator::
do_zlib_reset(::z_stream* strm)
  {
    int res = ::deflateReset(strm);
    if(res != Z_OK)
      this->do_zlib_throw_error("deflateReset", res);
  }

void
zlib_Deflator::
do_zlib_write_partial(int& res, ::z_stream* strm, int flush)
  {
    res = ::deflate(strm, flush);
    if(::rocket::is_none_of(res, { Z_OK, Z_STREAM_END, Z_BUF_ERROR }))
      this->do_zlib_throw_error("deflate", res);
  }

}  // namespace poseidon
