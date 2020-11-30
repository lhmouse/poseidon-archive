// This file is part of Poseidon.
// Copyleft 2020, LH_Mouse. All wrongs reserved.

#include "../precompiled.hpp"
#include "zlib.hpp"
#include "../util.hpp"

namespace poseidon {
namespace {

enum Format : int
  {
    format_deflate  =  0,
    format_gzip     = 16,
  };

struct Z_stream_deflate
final
  : public Z_stream
  {
    explicit
    Z_stream_deflate(Format format, int level = Z_DEFAULT_COMPRESSION)
      {
        // Create a deflate stream.
        int res = ::deflateInit2(&(this->m_strm), level, Z_DEFLATED,
                                 15 + format, 9, Z_DEFAULT_STRATEGY);
        if(res != Z_OK)
          this->do_throw_zlib_error("deflateInit2", res);
      }

    ~Z_stream_deflate()
      override
      {
        ::deflateEnd(&(this->m_strm));
      }

    void
    reset()
      noexcept override
      {
        int res = ::deflateReset(&(this->m_strm));
        ROCKET_ASSERT(res == Z_OK);

        this->m_obuf.clear();
      }

    void
    write(const void* data, size_t size)
      override
      {
        auto bptr = static_cast<const uint8_t*>(data);
        auto eptr = bptr + size;
        for(;;) {
          // Put some bytes into the stream.
          uint32_t navail = static_cast<uint32_t>(
                                ::rocket::min(eptr - bptr, INT_MAX));
          this->m_strm.next_in = bptr;
          this->m_strm.avail_in = navail;
          if(navail == 0)
            break;

          while(this->m_strm.avail_in != 0) {
            // Extend the output buffer so we never get `Z_BUF_ERROR`.
            this->do_reserve_output_buffer();

            int res = ::deflate(&(this->m_strm), Z_NO_FLUSH);
            if(res != Z_OK)
              this->do_throw_zlib_error("deflate", res);
          }
          bptr += navail;
        }
      }

    ::rocket::linear_buffer&
    synchronize()
      override
      {
        // Put nothing, but force `Z_SYNC_FLUSH`. If there is nothing
        // to do, `deflate()` will return `Z_BUF_ERROR`.
        this->m_strm.next_in = nullptr;
        this->m_strm.avail_in = 0;

        this->do_reserve_output_buffer();

        int res = ::deflate(&(this->m_strm), Z_SYNC_FLUSH);
        if((res != Z_OK) && (res != Z_BUF_ERROR))
          this->do_throw_zlib_error("deflate", res);

        // The output buffer is now ready for consumption.
        return this->do_update_output_buffer();
      }

    ::rocket::linear_buffer&
    finish()
      override
      {
        // Put nothing, but force `Z_FINISH`. If there is nothing to
        // do, `deflate()` will return `Z_STREAM_END`.
        this->m_strm.next_in = nullptr;
        this->m_strm.avail_in = 0;

        this->do_reserve_output_buffer();

        int res = ::deflate(&(this->m_strm), Z_FINISH);
        if(res != Z_STREAM_END)
          this->do_throw_zlib_error("deflate", res);

        // The output buffer is now ready for consumption.
        return this->do_update_output_buffer();
      }
  };

struct Z_stream_inflate
final
  : public Z_stream
  {
    explicit
    Z_stream_inflate(Format format)
      {
        // Create an inflate stream.
        int res = ::inflateInit2(&(this->m_strm), 15 + format);
        if(res != Z_OK)
          this->do_throw_zlib_error("inflateInit2", res);
      }

    ~Z_stream_inflate()
      override
      {
        ::inflateEnd(&(this->m_strm));
      }

    void
    reset()
      noexcept override
      {
        int res = ::inflateReset(&(this->m_strm));
        ROCKET_ASSERT(res == Z_OK);

        this->m_obuf.clear();
      }

    void
    write(const void* data, size_t size)
      override
      {
        auto bptr = static_cast<const uint8_t*>(data);
        auto eptr = bptr + size;
        for(;;) {
          // Put some bytes into the stream.
          uint32_t navail = static_cast<uint32_t>(
                                ::rocket::min(eptr - bptr, INT_MAX));
          this->m_strm.next_in = bptr;
          this->m_strm.avail_in = navail;
          if(navail == 0)
            break;

          while(this->m_strm.avail_in != 0) {
            // Extend the output buffer so we never get `Z_BUF_ERROR`.
            this->do_reserve_output_buffer();

            int res = ::inflate(&(this->m_strm), Z_NO_FLUSH);
            if(res != Z_OK)
              this->do_throw_zlib_error("inflate", res);
          }
          bptr += navail;
        }
      }

    ::rocket::linear_buffer&
    synchronize()
      override
      {
        // Put nothing, but force `Z_SYNC_FLUSH`. If there is nothing
        // to do, `inflate()` will return `Z_BUF_ERROR`.
        this->m_strm.next_in = nullptr;
        this->m_strm.avail_in = 0;

        this->do_reserve_output_buffer();

        int res = ::inflate(&(this->m_strm), Z_SYNC_FLUSH);
        if((res != Z_OK) && (res != Z_BUF_ERROR))
          this->do_throw_zlib_error("inflate", res);

        // The output buffer is now ready for consumption.
        return this->do_update_output_buffer();
      }

    ::rocket::linear_buffer&
    finish()
      override
      {
        // Put nothing, but force `Z_FINISH`. If there is nothing to
        // do, `inflate()` will return `Z_STREAM_END`.
        this->m_strm.next_in = nullptr;
        this->m_strm.avail_in = 0;

        this->do_reserve_output_buffer();

        int res = ::inflate(&(this->m_strm), Z_FINISH);
        if(res != Z_STREAM_END)
          this->do_throw_zlib_error("inflate", res);

        // The output buffer is now ready for consumption.
        return this->do_update_output_buffer();
      }
  };

}  // namespace

Z_stream::
~Z_stream()
  {
  }

void
Z_stream::
do_throw_zlib_error(const char* func, int err)
  const
  {
    const char* msg = this->m_strm.msg;
    if(!msg)
      msg = "[no message]";

    POSEIDON_THROW("zlib error: $1\n[`$2()` returned $3]",
                   msg, func, err);
  }

void
Z_stream::
do_reserve_output_buffer()
  {
    // Ensure there is some space in the output buffer.
    if((this->m_strm.next_out == reinterpret_cast<uint8_t*>(
               this->m_obuf.mut_end())) && (this->m_strm.avail_out > 64))
      return;

    uint32_t navail = static_cast<uint32_t>(this->m_obuf.reserve(256));
    this->m_strm.next_out = reinterpret_cast<uint8_t*>(
                                          this->m_obuf.mut_end());
    this->m_strm.avail_out = navail;
  }

::rocket::linear_buffer&
Z_stream::
do_update_output_buffer()
  {
    return this->m_obuf.accept(static_cast<size_t>(this->m_strm.next_out -
             reinterpret_cast<uint8_t*>(this->m_obuf.mut_end())));
  }

// Non-member functions
uptr<Z_stream>
create_deflator(int level)
  {
    return ::rocket::make_unique<Z_stream_deflate>(format_deflate, level);
  }

uptr<Z_stream>
create_inflator()
  {
    return ::rocket::make_unique<Z_stream_inflate>(format_deflate);
  }

uptr<Z_stream>
create_gzip_compressor(int level)
  {
    return ::rocket::make_unique<Z_stream_deflate>(format_gzip, level);
  }

uptr<Z_stream>
create_gzip_decompressor()
  {
    return ::rocket::make_unique<Z_stream_inflate>(format_gzip);
  }

}  // namespace poseidon
