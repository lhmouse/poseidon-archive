// This file is part of Poseidon.
// Copyleft 2020, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_CORE_ZLIB_DEFLATOR_HPP_
#define POSEIDON_CORE_ZLIB_DEFLATOR_HPP_

#include "../fwd.hpp"
#define ZLIB_CONST 1
#include <zlib.h>

namespace poseidon {

class zlib_Deflator
  : public ::asteria::Rcfwd<zlib_Deflator>
  {
  public:
    enum Format : uint8_t
      {
        format_deflate  = 0,
        format_raw      = 1,
        format_gzip     = 2,
      };

  private:
    ::z_stream m_strm;
    ::rocket::linear_buffer m_obuf;

  public:
    explicit
    zlib_Deflator(Format fmt, int level = Z_DEFAULT_COMPRESSION);

    ASTERIA_NONCOPYABLE_DESTRUCTOR(zlib_Deflator);

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

    zlib_Deflator&
    discard(size_t size)
      noexcept
      { return this->m_obuf.discard(size), *this;  }

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
    synchronize();

    // Terminates the stream.
    zlib_Deflator&
    finish();
  };

}  // namespace poseidon

#endif
