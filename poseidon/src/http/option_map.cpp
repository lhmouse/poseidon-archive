// This file is part of Poseidon.
// Copyleft 2020, LH_Mouse. All wrongs reserved.

#include "../precompiled.hpp"
#include "option_map.hpp"
#include "../util.hpp"

namespace poseidon {
namespace {

template<typename BucketT>
BucketT&
do_linear_probe(BucketT* bptr, BucketT* eptr, cow_string::shallow_type key, size_t hval)
  {
    // Find a bucket using linear probing.
    // We keep the load factor below 1.0 so there will always be some empty
    // buckets in the table.
    auto mptr = ::rocket::get_probing_origin(bptr, eptr, hval);
    auto qbkt = ::rocket::linear_probe(bptr, mptr, mptr, eptr,
            [&](const details_option_map::Bucket& r) { return r.key_equals(key);  });
    ROCKET_ASSERT(qbkt);
    return *qbkt;
  }

size_t
do_bucket_index(const ::rocket::cow_vector<details_option_map::Bucket>& stor,
                cow_string::shallow_type key, size_t hval)
  {
    ROCKET_ASSERT(!stor.empty());
    auto bptr = stor.data();
    auto& r = do_linear_probe(bptr, bptr + stor.size(), key, hval);
    return static_cast<size_t>(&r - bptr);
  }

enum : uint8_t
  {
    opt_ctype_control     = 0x01,  // control characters other than TAB
    opt_ctype_query_safe  = 0x02,  // usable in URL queries unquoted
    opt_ctype_http_tchar  = 0x04,  // token characters in HTTP headers
  };

constexpr uint8_t s_opt_ctype_table[128] =
  {
    0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01,
    0x01, 0x00, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01,
    0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01,
    0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01,
    0x00, 0x06, 0x00, 0x04, 0x06, 0x04, 0x04, 0x06,
    0x02, 0x02, 0x06, 0x04, 0x02, 0x06, 0x06, 0x02,
    0x06, 0x06, 0x06, 0x06, 0x06, 0x06, 0x06, 0x06,
    0x06, 0x06, 0x02, 0x02, 0x00, 0x00, 0x00, 0x02,
    0x02, 0x06, 0x06, 0x06, 0x06, 0x06, 0x06, 0x06,
    0x06, 0x06, 0x06, 0x06, 0x06, 0x06, 0x06, 0x06,
    0x06, 0x06, 0x06, 0x06, 0x06, 0x06, 0x06, 0x06,
    0x06, 0x06, 0x06, 0x00, 0x00, 0x00, 0x04, 0x06,
    0x04, 0x06, 0x06, 0x06, 0x06, 0x06, 0x06, 0x06,
    0x06, 0x06, 0x06, 0x06, 0x06, 0x06, 0x06, 0x06,
    0x06, 0x06, 0x06, 0x06, 0x06, 0x06, 0x06, 0x06,
    0x06, 0x06, 0x06, 0x00, 0x04, 0x00, 0x06, 0x01,
  };

constexpr
uint8_t
do_get_opt_ctype(char c)
  noexcept
  { return (uint8_t(c) < 128) ? s_opt_ctype_table[uint8_t(c)] : 0;  }

constexpr
bool
do_is_opt_ctype(char c, uint8_t mask)
  noexcept
  { return do_get_opt_ctype(c) & mask;  }

constexpr char s_xdigits[] = "00112233445566778899AaBbCcDdEeFf";

tinyfmt&
do_encode_query(tinyfmt& fmt, const cow_string& str)
  {
    const char* bp = str.c_str();
    const char* ep = bp + str.size();
    for(;;) {
      // Get the first sequence of safe characters.
      const char* mp = ::std::find_if(bp, ep,
             [&](char ch) { return !do_is_opt_ctype(ch, opt_ctype_query_safe);  });
      if(mp != bp)
        fmt.putn(bp, static_cast<size_t>(mp - bp));

      bp = mp;
      if(bp == ep)
        break;

      // Encode one unsafe character.
      uint32_t val = static_cast<uint8_t>(*(bp++));
      if(val == ' ') {
        fmt.putc('+');
        continue;
      }

      char seq[3];
      seq[0] = '%';
      seq[1] = s_xdigits[(val >> 3) & 0x1E];
      seq[2] = s_xdigits[(val << 1) & 0x1E];
      fmt.putn(seq, sizeof(seq));
    }
    return fmt;
  }

cow_string&
do_decode_query(cow_string& str, const char* bptr, const char* eptr)
  {
    str.clear();
    const char* bp = bptr;
    while(bp != eptr) {
      // Process non-percent sequences.
      uint32_t val = static_cast<uint8_t>(*(bp++));
      if(val == '+') {
        str += ' ';
        continue;
      }
      else if(val != '%') {
        str += static_cast<char>(val);
        continue;
      }

      // Parse two hexadecimal digits following the percent sign.
      // Note `do_xdigit_value()` shall throw an exception if `*bp` is zero
      // i.e. when `bp == eptr`.
      for(uint32_t k = 0;  k != 2;  ++k) {
        auto dp = static_cast<const char*>(::std::memchr(s_xdigits, *(bp++), 32));
        if(!dp)
          POSEIDON_THROW("Invalid hexadecimal digit: $1", bp[-1]);

        val = val << 4 | static_cast<uint32_t>(dp - s_xdigits) >> 1;
      }
      str += static_cast<char>(val);
    }
    return str;
  }

} // namespace

Option_Map::
~Option_Map()
  {
  }

details_option_map::Range<const cow_string>
Option_Map::
do_range_hint(cow_string::shallow_type key, size_t hval)
  const noexcept
  {
    if(this->m_stor.empty())
      return { };

    size_t index = do_bucket_index(this->m_stor, key, hval);
    return this->m_stor[index].range();
  }

details_option_map::Range<cow_string>
Option_Map::
do_mut_range_hint(cow_string::shallow_type key, size_t hval)
  {
    if(this->m_stor.empty())
      return { };

    size_t index = do_bucket_index(this->m_stor, key, hval);
    if(this->m_stor[index].count() == 0)
      return { };

    return this->m_stor.mut(index).mut_range();
  }

size_t
Option_Map::
do_erase_hint(cow_string::shallow_type key, size_t hval)
  {
    if(this->m_stor.empty())
      return 0;

    size_t index = do_bucket_index(this->m_stor, key, hval);
    size_t count = this->m_stor[index].count();
    if(count == 0)
      return 0;

    // Clear the bucket.
    details_option_map::Bucket temp;
    auto bptr = this->m_stor.mut_data();
    auto eptr = bptr + this->m_stor.size();

    bptr[index].reset();
    this->m_nbkt -= 1;

    // Reallocate buckets that follow `bkt`.
    ::rocket::linear_probe(
      bptr,
      bptr + index,
      bptr + index + 1,
      eptr,
      [&](details_option_map::Bucket& r) {
        // Make the old bucket empty.
        temp = ::std::move(r);
        r.reset();

        // Find a new bucket for the name using linear probing.
        // Uniqueness has already been implied for all elements, so there is no
        // need to check for collisions.
        auto mptr = ::rocket::get_probing_origin(bptr, eptr, r.hash());
        auto qbkt = ::rocket::linear_probe(bptr, mptr, mptr, eptr,
                      [&](const details_option_map::Bucket&) { return false;  });
        ROCKET_ASSERT(qbkt);

        // Insert it back.
        *qbkt = ::std::move(temp);
        return false;
      });

    // Return the number of elements that have been erased.
    return count;
  }

details_option_map::Bucket&
Option_Map::
do_reserve(const cow_string& key, size_t hval)
  {
    auto bptr = this->m_stor.mut_data();
    auto eptr = bptr + this->m_stor.size();

    if(bptr != eptr) {
      // Use any existent bucket if an equivalent key has been found.
      auto& bkt = do_linear_probe(bptr, eptr, sref(key), hval);
      if(bkt)
        return bkt;

      // If the load factor is below 0.5, use this empty bucket.
      if(this->m_nbkt < static_cast<size_t>(eptr - bptr) / 2)
        return bkt;
    }

    // Allocate a new table.
    ::rocket::cow_vector<details_option_map::Bucket> stor(this->m_nbkt * 3 | 17);
    bptr = stor.mut_data();
    eptr = bptr + stor.size();

    // Move-assign buckets into the new table.
    ::std::for_each(this->m_stor.mut_begin(), this->m_stor.mut_end(),
      [&](details_option_map::Bucket& r) {
        if(r.count())
          do_linear_probe(bptr, eptr, sref(r.key()), r.hash()) = ::std::move(r);
      });

    // Set up the new table.
    this->m_stor.swap(stor);

    // Find a bucket for the given key.
    return do_linear_probe(bptr, eptr, sref(key), hval);
  }

cow_string&
Option_Map::
do_open_hint(const cow_string& key, size_t hval)
  {
    // Search for an existent key.
    auto& bkt = this->do_reserve(key, hval);
    if(ROCKET_EXPECT(bkt))
      return bkt.mut_scalar();

    // Initialize this new bucket.
    bkt.set_key(key);
    auto& str = bkt.mut_scalar();
    this->m_nbkt++;
    return str;
  }

cow_string&
Option_Map::
do_append_hint(const cow_string& key, size_t hval)
  {
    // Search for an existent key.
    auto& bkt = this->do_reserve(key, hval);
    if(ROCKET_EXPECT(bkt))
      return bkt.mut_array().emplace_back();

    // Initialize this new bucket.
    bkt.set_key(key);
    auto& arr = bkt.mut_array();
    this->m_nbkt++;
    return arr.emplace_back();
  }

tinyfmt&
Option_Map::
print(tinyfmt& fmt)
  const
  {
    fmt << "{";
    for(const auto& bkt : this->m_stor) {
      for(auto r = bkt.range();  r.first != r.second;  r.first++) {
        // Indent each line a bit.
        fmt << "\n  ";

        // Print the quoted key and value.
        fmt << ::asteria::quote(bkt.key());
        fmt << " = ";
        fmt << ::asteria::quote(*(r.first));
      }
    }
    fmt << "\n}\n";
    return fmt;
  }

tinyfmt&
Option_Map::
print_url_query(tinyfmt& fmt)
  const
  {
    size_t count = SIZE_MAX;
    for(const auto& bkt : this->m_stor) {
      for(auto r = bkt.range();  r.first != r.second;  r.first++) {
        // Separate elements with ampersands.
        if(++count)
          fmt << '&';

        // Encode keys and values.
        do_encode_query(fmt, bkt.key());
        fmt << '=';
        do_encode_query(fmt, *(r.first));
      }
    }
    return fmt;
  }

Option_Map&
Option_Map::
parse_url_query(const cow_string& str)
  {
    // Destroy existent data.
    // If this function fails, the contents of `*this` are undefined.
    this->clear();

    // Ensure the string doesn't contain blank or control characters.
    if(::rocket::any_of(str,
             [&](char ch) { return ::rocket::is_any_of(ch, {' ', '\t'}) ||
                                   do_is_opt_ctype(ch, opt_ctype_control);  }))
      POSEIDON_THROW("Invalid character in URL query string: $1", str);

    // Why pointers? Why not iterators?
    // We assume that the string is terminated by a null character, which
    // simplifies a lot of checks below. But dereferencing the past-the-end
    // iterator results in undefined behavior. On the other hand, as string
    // is a container whose elements are consecutive, dereferencing the
    // past-the-end pointer always yields a read-only but valid reference
    // to the null terminator and is not undefined behavior.
    const char* bptr = str.c_str();
    const char* const eptr = bptr + str.size();
    const char* mptr;

    for(;;) {
      // Skip separators.
      bptr = ::std::find_if(bptr, eptr, [&](char ch) { return ch != '&';  });
      if(bptr == eptr)
        break;

      // A key is terminated by either delimiter.
      mptr = ::std::find_if(bptr, eptr,
                 [&](char ch) { return ::rocket::is_any_of(ch, {'=', '&'});  });

      // Decode the key.
      cow_string key;
      do_decode_query(key, bptr, mptr);
      bptr = mptr;

      // Note a value may contain equals signs.
      if(bptr[0] == '=')
        mptr = ::std::find(++bptr, eptr, '&');

      // Decode the value.
      do_decode_query(this->append(key), bptr, mptr);
      bptr = mptr;
    }
    return *this;
  }

tinyfmt&
Option_Map::
print_http_header(tinyfmt& fmt)
  const
  {
    // Write non-options first.
    size_t count = SIZE_MAX;
    for(auto r = this->range(::rocket::sref(""));  r.first != r.second;  r.first++) {
      // Separate fields with semicolons.
      if(++count)
        fmt << "; ";

      // All non-options must not contain control characters.
      if(::rocket::any_of(*(r.first),
                 [&](char ch) { return do_is_opt_ctype(ch, opt_ctype_control);  }))
        POSEIDON_THROW("Invalid character in HTTP header: $1", *(r.first));

      // Write the non-option string unquoted.
      fmt << *(r.first);
    }

    // Write options.
    for(const auto& bkt : this->m_stor) {
      if(bkt.key().empty())
        continue;

      // The key must be a valid token.
      if(::rocket::any_of(bkt.key(),
                 [&](char ch) { return !do_is_opt_ctype(ch, opt_ctype_http_tchar);  }))
        POSEIDON_THROW("Invalid HTTP header token: $1", bkt.key());

      for(auto r = bkt.range();  r.first != r.second;  r.first++) {
        // The value must contain no control characters other than TAB.
        if(::rocket::any_of(*(r.first),
                 [&](char ch) { return do_is_opt_ctype(ch, opt_ctype_control);  }))
          POSEIDON_THROW("Invalid HTTP header value: $1", *(r.first));

        // Separate fields with semicolons.
        if(++count)
          fmt << "; ";

        // Write the key unquoted.
        fmt << bkt.key();

        // If the value is empty, no equals sign appears.
        if(r.first->empty())
          continue;

        // Search for characters that must be escaped.
        // If no character needs escaping, write it without double quotes.
        const char* bp = r.first->c_str();
        const char* ep = bp + r.first->size();
        if(::std::all_of(bp, ep,
                 [&](char ch) { return do_is_opt_ctype(ch, opt_ctype_http_tchar);  })) {
          fmt << '=' << *(r.first);
          continue;
        }

        // Escape the value.
        // There shall be no bad whitespace (BWS) on either side of the equals sign.
        fmt << "=\"";
        for(;;) {
          // Get the first sequence of safe characters.
          const char* mp = ::std::find_if(bp, ep,
                 [&](char ch) { return ::rocket::is_any_of(ch, {'\"', '\\'});  });
          if(mp != bp)
            fmt.putn(bp, static_cast<size_t>(mp - bp));

          bp = mp;
          if(bp == ep)
            break;

          // Escape this character.
          char seq[2];
          seq[0] = '\\';
          seq[1] = *(bp++);
          fmt.putn(seq, sizeof(seq));
        }
        fmt << '\"';
      }
    }
    return fmt;
  }

Option_Map&
Option_Map::
parse_http_header(size_t* comma_opt, const cow_string& str, size_t nonopts)
  {
    // Destroy existent data.
    // If this function fails, the contents of `*this` are undefined.
    this->clear();

    // Ensure the string doesn't contain control characters.
    if(::rocket::any_of(str,
              [&](char ch) { return do_is_opt_ctype(ch, opt_ctype_control);  }))
      POSEIDON_THROW("Invalid character in HTTP header: $1", str);

    // Why pointers? Why not iterators?
    // We assume that the string is terminated by a null character, which simplifies
    // a lot of checks below. But dereferencing the past-the-end iterator results in
    // undefined behavior. On the other hand, as string is a container whose elements
    // are contiguous, dereferencing the past-the-end pointer always yields a
    // read-only but valid reference to the null terminator and is not undefined
    // behavior.
    const char* bp = str.c_str();
    const char* ep = bp + str.size();
    const char* mp;

    // Adjust the start position.
    if(comma_opt)
      bp = ::std::addressof(str.at(*comma_opt));

    // Skip leading blank characters and commas.
    // Note that leading commas are always skipped despite `comma_opt`.
    bp = ::std::find_if(bp, ep,
              [&](char ch) { return ::rocket::is_none_of(ch, {' ', '\t', ','});  });
    if(bp == ep)
      return *this;

    size_t count = SIZE_MAX;
    cow_string key;
    for(;;) {
      // Skip leading blank characters.
      bp = ::std::find_if(bp, ep,
              [&](char ch) { return ::rocket::is_none_of(ch, {' ', '\t'});  });
      if(bp == ep)
        break;

      if(comma_opt && (*bp == ','))
        break;

      // Note that the storage of `key` may be reused.
      count++;
      key.clear();

      // Skip empty fields.
      if(*bp == ';') {
        ++bp;
        continue;
      }

      if(count >= nonopts) {
        // Get a key, which shall be a token.
        mp = ::std::find_if(bp, ep,
              [&](char ch) { return !do_is_opt_ctype(ch, opt_ctype_http_tchar);  });
        if(mp == bp)
          POSEIDON_THROW("Invalid HTTP header (token expected): $1", str);

        key.append(bp, mp);
        bp = mp;

        // Skip trailing blank characters.
        bp = ::std::find_if(bp, ep,
              [&](char ch) { return ::rocket::is_none_of(ch, {' ', '\t'});  });

        // If the key is terminated by a comma, semicolon, or the end of
        // input, accept a key with an empty value.
        if((bp == ep) || ::rocket::is_any_of(*bp, {',', ';'})) {
          this->append(key);
          continue;
        }

        // Otherwise, an equals sign shall follow.
        if(*bp != '=')
          POSEIDON_THROW("Invalid HTTP header (`=` expected): $1", str);

        // Skip trailing blank characters.
        bp = ::std::find_if(++bp, ep,
              [&](char ch) { return ::rocket::is_none_of(ch, {' ', '\t'});  });
      }

      // Get a value.
      auto& val = this->append(key);
      if(*bp == '\"') {
        // If the value starts with a double quote, it shall be a quoted string.
        ++bp;
        for(;;) {
          mp = ::std::find_if(bp, ep,
              [&](char ch) { return ::rocket::is_any_of(ch, {'\\', '\"'});  });
          if(mp != bp)
            val.append(bp, mp);
          bp = mp;

          if(bp == ep)
            POSEIDON_THROW("Invalid HTTP header (missing `\"`): $1", str);

          if(*bp == '\"')
            break;

          // Unescape this character.
          if(++bp == ep)
            POSEIDON_THROW("Invalid HTTP header (dangling `\\` at the end): $1", str);
          val.push_back(*(bp++));
        }
        ++bp;
        continue;
      }

      // Otherwise, the value is copied verbatim up to the first comma or semicolon.
      mp = bp;
      while((mp != ep) && (*mp != ';') && !(comma_opt && (*mp == ','))) {
        // Record the start of this whitespace sequence in `mptr`, but don't append
        // it right now.
        char ch = *(mp++);
        if(::rocket::is_any_of(ch, {' ', '\t'}))
          continue;

        // Append the non-whitespace character along with any preceding whitespace
        // characters.
        val.append(bp, mp);
        bp = mp;
      }
    }

    // Output the end position.
    if(comma_opt)
      *comma_opt = static_cast<size_t>(bp - str.data());
    return *this;
  }

} // namespace poseidon
