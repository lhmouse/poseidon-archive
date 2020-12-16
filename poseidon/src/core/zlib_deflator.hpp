// This file is part of Poseidon.
// Copyleft 2020, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_CORE_ZLIB_DEFLATOR_HPP_
#define POSEIDON_CORE_ZLIB_DEFLATOR_HPP_

#include "../fwd.hpp"
#include "../details/zlib_stream_common.hpp"

namespace poseidon {

class zlib_Deflator
  : public ::asteria::Rcfwd<zlib_Deflator>,
    private details_zlib_stream_common::zlib_Stream_Common
  {
  public:
    enum Format : uint8_t
      {
        format_deflate  = 0,
        format_raw      = 1,
        format_gzip     = 2,
      };

  public:
    explicit
    zlib_Deflator(Format fmt, int level = Z_DEFAULT_COMPRESSION);

  public:
    ASTERIA_NONCOPYABLE_DESTRUCTOR(zlib_Deflator);

    // Gets the output buffer.
    const ::rocket::linear_buffer&
    output_buffer()
      const noexcept
      { return this->m_obuf;  }

    ::rocket::linear_buffer&
    output_buffer()
      noexcept
      { return this->m_obuf;  }

    // Resets internal states and clears the output buffer.
    // Unprocessed data are discarded.
    zlib_Deflator&
    reset()
      noexcept;

    // Puts some data for compression/decompression.
    zlib_Deflator&
    write(const char* data, size_t size);

    // Synchronizes output to a byte boundary. This causes 4 extra
    // bytes (00 00 FF FF) to be appended to the output buffer.
    zlib_Deflator&
    flush();

    // Terminates the stream.
    zlib_Deflator&
    finish();
  };

}  // namespace poseidon

#endif
