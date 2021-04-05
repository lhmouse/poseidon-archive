// This file is part of Poseidon.
// Copyleft 2020, LH_Mouse. All wrongs reserved.

#include "../precompiled.hpp"
#include "zlib_stream_common.hpp"
#include "../utils.hpp"

namespace poseidon {
namespace details_zlib_stream_common {

zlib_Stream_Common::
~zlib_Stream_Common()
  {
    // Abuse the opaque pointer for the destructor.
    auto dtor = reinterpret_cast<destructor*>(this->m_strm.opaque);
    if(dtor)
      (*dtor)(&(this->m_strm));
  }

void
zlib_Stream_Common::
do_reserve_output_buffer()
  {
    // Ensure there is enough space in the output buffer.
    size_t navail = this->m_obuf.reserve(64);
    this->m_strm.next_out = reinterpret_cast<uint8_t*>(this->m_obuf.mut_end());
    this->m_strm.avail_out = static_cast<uint32_t>(navail);
  }

void
zlib_Stream_Common::
do_update_output_buffer() noexcept
  {
    // Consume output bytes, if any.
    auto pbase = reinterpret_cast<const uint8_t*>(this->m_obuf.end());
    this->m_obuf.accept(static_cast<size_t>(this->m_strm.next_out - pbase));
  }

void
zlib_Stream_Common::
do_zlib_throw_error(const char* func, int err) const
  {
    // Note this field is always initialized.
    const char* msg = this->m_strm.msg;
    if(!msg)
      msg = "[no message]";

    POSEIDON_THROW("zlib error: $1\n[`$2()` returned $3]", msg, func, err);
  }

void
zlib_Stream_Common::
do_construct(int level, int wbits, destructor* dtor)
  {
    // Initialize requred fields.
#ifdef ROCKET_DEBUG
    ::std::memset(&(this->m_strm), 0xEE, sizeof(this->m_strm));
#endif
    this->m_strm.next_in = nullptr;
    this->m_strm.avail_in = 0;
    this->m_strm.zalloc = nullptr;
    this->m_strm.zfree = nullptr;

    // Note exception safety.
    this->m_strm.opaque = nullptr;
    this->do_zlib_construct(&(this->m_strm), level, wbits);
    this->m_strm.opaque = reinterpret_cast<void*>(dtor);
  }

void
zlib_Stream_Common::
do_reset()
  {
    // Clear everything.
    this->do_zlib_reset(&(this->m_strm));

    this->m_strm.next_out = nullptr;
    this->m_strm.avail_out = 0;
    this->m_obuf.clear();
  }

void
zlib_Stream_Common::
do_write(const char* data, size_t size)
  {
    // Set up the read pointer.
    const auto end_in = reinterpret_cast<const uint8_t*>(data + size);
    this->m_strm.next_in = end_in - size;

    int res;
    do {
      // The stupid zlib library uses a 32-bit integer for number of bytes.
      this->m_strm.avail_in = static_cast<uint32_t>(
                     ::rocket::min(end_in - this->m_strm.next_in, INT_MAX));

      // Extend the output buffer first so we never get `Z_BUF_ERROR`.
      this->do_reserve_output_buffer();
      this->do_zlib_write_partial(res, &(this->m_strm), Z_NO_FLUSH);
      this->do_update_output_buffer();
    }
    while(res == Z_OK);
  }

void
zlib_Stream_Common::
do_flush()
  {
    // Put nothing, but force `Z_SYNC_FLUSH`.
    this->m_strm.next_in = nullptr;
    this->m_strm.avail_in = 0;

    int res;
    do {
      // Flush all output data into the output buffer.
      this->do_reserve_output_buffer();
      this->do_zlib_write_partial(res, &(this->m_strm), Z_SYNC_FLUSH);
      this->do_update_output_buffer();
    }
    while(res == Z_OK);
  }

void
zlib_Stream_Common::
do_finish()
  {
    // Put nothing, but force `Z_FINISH`.
    this->m_strm.next_in = nullptr;
    this->m_strm.avail_in = 0;

    int res;
    do {
      // Flush all output data into the output buffer.
      this->do_reserve_output_buffer();
      this->do_zlib_write_partial(res, &(this->m_strm), Z_FINISH);
      this->do_update_output_buffer();
    }
    while(res == Z_OK);
  }

}  // namespace details_zlib_stream_common
}  // namespace poseidon
