// This file is part of Poseidon.
// Copyleft 2020, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_CORE_ZLIB_INFLATOR_HPP_
#define POSEIDON_CORE_ZLIB_INFLATOR_HPP_

#include "../fwd.hpp"
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
    ::z_stream m_strm = { };
    ::rocket::linear_buffer m_obuf;

  public:
    explicit
    zlib_Inflator(Format fmt);

    ASTERIA_NONCOPYABLE_DESTRUCTOR(zlib_Inflator);

  private:
    [[noreturn]] inline
    void
    do_throw_zlib_error(const char* func, int err)
      const;

    inline
    void
    do_reserve_output_buffer();

    inline
    void
    do_update_output_buffer()
      noexcept;

  public:
    // Gets the output buffer.
    const char*
    output_data()
      const noexcept
      { return this->m_obuf.begin();  }

    size_t
    output_size()
      const noexcept
      { return this->m_obuf.size();  }

    zlib_Inflator&
    output_consume(size_t size)
      noexcept
      { return this->m_obuf.discard(size), *this;  }

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
    synchronize();

    // Terminates the stream.
    zlib_Inflator&
    finish();
  };

}  // namespace poseidon

#endif
