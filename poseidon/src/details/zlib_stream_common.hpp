// This file is part of Poseidon.
// Copyleft 2020, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_DETAILS_ZLIB_STREAM_COMMON_HPP_
#define POSEIDON_DETAILS_ZLIB_STREAM_COMMON_HPP_

#include "../fwd.hpp"
#define ZLIB_CONST 1
#include <zlib.h>

namespace poseidon {
namespace details_zlib_stream_common {

class zlib_Stream_Common
  : public ::asteria::Rcfwd<zlib_Stream_Common>
  {
  protected:
    using destructor  = int (::z_stream* strm);

  private:
    ::z_stream m_strm;
    ::rocket::linear_buffer m_obuf;

  protected:
    zlib_Stream_Common() noexcept
      = default;

  private:
    inline void
    do_reserve_output_buffer();

    inline void
    do_update_output_buffer() noexcept;

  protected:
    [[noreturn]] void
    do_zlib_throw_error(const char* func, int err) const;

    // These are callback functions.
    virtual void
    do_zlib_construct(::z_stream* strm, int level, int wbits)
      = 0;

    virtual void
    do_zlib_reset(::z_stream* strm)
      = 0;

    virtual void
    do_zlib_write_partial(int& res, ::z_stream* strm, int flush)
      = 0;

    // Initializes internal states.
    void
    do_construct(int level, int wbits, destructor* dtor);

    // Resets internal states and clears the output buffer.
    // Unprocessed data are discarded.
    void
    do_reset();

    // Puts some data for compression/decompression.
    void
    do_write(const char* data, size_t size);

    // Synchronizes output to a byte boundary. This causes 4 extra
    // bytes (00 00 FF FF) to be appended to the output buffer.
    void
    do_flush();

    // Terminates the stream.
    void
    do_finish();

  public:
    ASTERIA_NONCOPYABLE_DESTRUCTOR(zlib_Stream_Common);

    // Gets the output buffer.
    const ::rocket::linear_buffer&
    output_buffer() const noexcept
      { return this->m_obuf;  }

    ::rocket::linear_buffer&
    output_buffer() noexcept
      { return this->m_obuf;  }
  };

}  // namespace details_zlib_stream_common
}  // namespace poseidon

#endif
