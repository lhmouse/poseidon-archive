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

    // Create a deflate stream.
    // Note this must be the last operation in each constructor.
    int res = ::deflateInit2(this->m_strm, rlevel, Z_DEFLATED, wbits, 9,
                             Z_DEFAULT_STRATEGY);
    if(res != Z_OK)
      this->do_throw_zlib_error("deflateInit2", res);

    this->do_set_destructor(::deflateEnd);
  }

zlib_Deflator::
~zlib_Deflator()
  {
  }

zlib_Deflator&
zlib_Deflator::
reset()
  noexcept
  {
    int res = ::deflateReset(this->m_strm);
    ROCKET_ASSERT(res == Z_OK);

    this->m_strm->next_out = nullptr;
    this->m_strm->avail_out = 0;
    this->m_obuf.clear();
    return *this;
  }

zlib_Deflator&
zlib_Deflator::
write(const char* data, size_t size)
  {
    // Set up the read pointer.
    const auto eptr = reinterpret_cast<const uint8_t*>(data + size);
    this->m_strm->next_in = eptr - size;
    for(;;) {
      // The stupid zlib library uses a 32-bit integer for number of bytes.
      this->m_strm->avail_in = static_cast<uint32_t>(
                ::rocket::min(eptr - this->m_strm->next_in, INT32_MAX));
      if(this->m_strm->avail_in == 0)
        break;

      // Extend the output buffer so we never get `Z_BUF_ERROR`.
      this->do_reserve_output_buffer();
      int res = ::deflate(this->m_strm, Z_NO_FLUSH);
      if(res != Z_OK)
        this->do_throw_zlib_error("deflate", res);

      this->do_update_output_buffer();
    }
    return *this;
  }

zlib_Deflator&
zlib_Deflator::
flush()
  {
    // Put nothing, but force `Z_SYNC_FLUSH`.
    this->m_strm->next_in = nullptr;
    this->m_strm->avail_in = 0;

    // If there is nothing to do, `deflate()` will return `Z_BUF_ERROR`.
    this->do_reserve_output_buffer();
    int res = ::deflate(this->m_strm, Z_SYNC_FLUSH);
    if((res != Z_OK) && (res != Z_BUF_ERROR))
      this->do_throw_zlib_error("deflate", res);

    this->do_update_output_buffer();
    return *this;
  }

zlib_Deflator&
zlib_Deflator::
finish()
  {
    // Put nothing, but force `Z_FINISH`.
    this->m_strm->next_in = nullptr;
    this->m_strm->avail_in = 0;

    // If there is nothing to do, `deflate()` will return `Z_STREAM_END`.
    this->do_reserve_output_buffer();
    int res = ::deflate(this->m_strm, Z_FINISH);
    if(res != Z_STREAM_END)
      this->do_throw_zlib_error("deflate", res);

    this->do_update_output_buffer();
    return *this;
  }

}  // namespace poseidon
