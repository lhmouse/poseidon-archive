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
    ::rocket::linear_buffer&
    do_update_output_buffer();

  public:
    // Resets internal states and clears the output buffer.
    // Unprocessed data are discarded.
    void
    reset()
      noexcept;

    // Puts some data for compression/decompression.
    void
    write(const void* data, size_t size);

    // Synchronizes output to a byte boundary and returns the output
    // buffer, which may be partially consumed. This causes an extra
    // 4 bytes (00 00 FF FF) to be appended to the output buffer.
    ::rocket::linear_buffer&
    synchronize();

    // Terminates the stream returns the output buffer.
    ::rocket::linear_buffer&
    finish();
  };

}  // namespace poseidon

#endif
