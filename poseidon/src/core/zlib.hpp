// This file is part of Poseidon.
// Copyleft 2020, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_CORE_ZLIB_HPP_
#define POSEIDON_CORE_ZLIB_HPP_

#include "../fwd.hpp"
#define ZLIB_CONST 1
#include <zlib.h>

namespace poseidon {

class Z_stream
  : public ::asteria::Rcfwd<Z_stream>
  {
  protected:
    ::z_stream m_strm = { };
    ::rocket::linear_buffer m_obuf;

  public:
    ASTERIA_NONCOPYABLE_DESTRUCTOR(Z_stream);

  protected:
    Z_stream()
      = default;

    [[noreturn]]
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
    virtual
    void
    reset()
      noexcept
      = 0;

    // Puts some data for compression/decompression.
    virtual
    void
    write(const void* data, size_t size)
      = 0;

    // Synchronizes output to a byte boundary and returns the output
    // buffer, which may be partially consumed. This causes an extra
    // 4 bytes (00 00 FF FF) to be appended to the output buffer.
    virtual
    ::rocket::linear_buffer&
    synchronize()
      = 0;

    // Terminates the stream returns the output buffer.
    virtual
    ::rocket::linear_buffer&
    finish()
      = 0;
  };

// Create a compressor/decompressor using the deflate format.
uptr<Z_stream>
create_deflator(int level = Z_DEFAULT_COMPRESSION);

uptr<Z_stream>
create_inflator();

// Create a compressor/decompressor using the gzip format.
uptr<Z_stream>
create_gzip_compressor(int level = Z_DEFAULT_COMPRESSION);

uptr<Z_stream>
create_gzip_decompressor();

}  // namespace poseidon

#endif
