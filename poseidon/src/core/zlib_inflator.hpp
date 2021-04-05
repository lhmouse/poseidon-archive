// This file is part of Poseidon.
// Copyleft 2020, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_CORE_ZLIB_INFLATOR_HPP_
#define POSEIDON_CORE_ZLIB_INFLATOR_HPP_

#include "../fwd.hpp"
#include "../details/zlib_stream_common.hpp"

namespace poseidon {

class zlib_Inflator
  : public ::asteria::Rcfwd<zlib_Inflator>,
    private details_zlib_stream_common::zlib_Stream_Common
  {
  public:
    enum Format : uint8_t
      {
        format_deflate  = 0,
        format_raw      = 1,
        format_gzip     = 2,
        format_auto     = 3,  // deflate or gzip
      };

  public:
    explicit
    zlib_Inflator(Format fmt);

  private:
    // These are overridden callbacks.
    void
    do_zlib_construct(::z_stream* strm, int level, int wbits) final;

    void
    do_zlib_reset(::z_stream* strm) final;

    void
    do_zlib_write_partial(int& res, ::z_stream* strm, int flush) final;

  public:
    ASTERIA_NONCOPYABLE_DESTRUCTOR(zlib_Inflator);

    // Gets the output buffer.
    using zlib_Stream_Common::output_buffer;

    // Resets internal states and clears the output buffer.
    // Unprocessed data are discarded.
    zlib_Inflator&
    reset()
      { return this->do_reset(), *this;  }

    // Puts some data for compression/decompression.
    zlib_Inflator&
    write(const char* data, size_t size)
      { return this->do_write(data, size), *this;  }

    // Synchronizes output to a byte boundary. This causes 4 extra
    // bytes (00 00 FF FF) to be appended to the output buffer.
    zlib_Inflator&
    flush()
      { return this->do_flush(), *this;  }

    // Terminates the stream.
    zlib_Inflator&
    finish()
      { return this->do_finish(), *this;  }
  };

}  // namespace poseidon

#endif
