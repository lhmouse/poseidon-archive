// This file is part of Poseidon.
// Copyleft 2020, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_CORE_ZLIB_INFLATOR_HPP_
#define POSEIDON_CORE_ZLIB_INFLATOR_HPP_

#include "../fwd.hpp"
#include "../details/zlib_stream_common.hpp"
#define ZLIB_CONST 1
#include <zlib.h>

namespace poseidon {

class zlib_Inflator
  : public ::asteria::Rcfwd<zlib_Inflator>
  {
  public:
    enum Format : uint8_t
      {
        format_deflate  = 0,
        format_raw      = 1,
        format_gzip     = 2,
        format_auto     = 3,  // deflate or gzip
      };

  private:
    details_zlib_stream_common::zlib_Stream_Common m_zlib;

  public:
    explicit
    zlib_Inflator(Format fmt);

    ASTERIA_NONCOPYABLE_DESTRUCTOR(zlib_Inflator);

  public:
    // Gets the output buffer.
    const ::rocket::linear_buffer&
    output_buffer()
      const noexcept
      { return this->m_zlib.obuf;  }

    ::rocket::linear_buffer&
    output_buffer()
      noexcept
      { return this->m_zlib.obuf;  }

    // Resets internal states and clears the output buffer.
    // Unprocessed data are discarded.
    zlib_Inflator&
    reset()
      noexcept;

    // Puts some data for compression/decompression.
    zlib_Inflator&
    write(const char* data, size_t size);

    // Causes as many bytes as possible to be flushed into the
    // output buffer, which may be consumed immediately.
    zlib_Inflator&
    flush();

    // Terminates the stream.
    zlib_Inflator&
    finish();
  };

}  // namespace poseidon

#endif
