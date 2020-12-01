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
    write(const char* data, size_t size);

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
