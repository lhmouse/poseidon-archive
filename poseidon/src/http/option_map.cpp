// This file is part of Poseidon.
// Copyleft 2020, LH_Mouse. All wrongs reserved.

#include "../precompiled.hpp"
#include "option_map.hpp"
#include "../utilities.hpp"

namespace poseidon {
namespace {

enum : uint8_t
  {
    opt_ctype_control     = 0x01,  // control character other than TAB
    opt_ctype_query_safe  = 0x02,  // usable in URL queries unquoted
  };

constexpr uint8_t s_opt_ctype_table[128] =
  {
    0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01,
    0x01, 0x00, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01,
    0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01,
    0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01,
    0x00, 0x02, 0x00, 0x00, 0x02, 0x00, 0x00, 0x02,
    0x02, 0x02, 0x02, 0x00, 0x02, 0x02, 0x02, 0x02,
    0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02,
    0x02, 0x02, 0x02, 0x02, 0x00, 0x00, 0x00, 0x02,
    0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02,
    0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02,
    0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02,
    0x02, 0x02, 0x02, 0x00, 0x00, 0x00, 0x00, 0x02,
    0x00, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02,
    0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02,
    0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02,
    0x02, 0x02, 0x02, 0x00, 0x00, 0x00, 0x02, 0x01,
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

tinyfmt&
do_encode_query(tinyfmt& fmt, const cow_string& str)
  {
    const char* bptr = str.c_str();
    const char* const eptr = bptr + str.size();

    for(;;) {
      // Get the first sequence of safe characters.
      auto mptr = ::std::find_if(bptr, eptr,
                      [&](char ch) { return !do_is_opt_ctype(ch, opt_ctype_query_safe);  });

      if(mptr != bptr)
        fmt.putn(bptr, static_cast<size_t>(mptr - bptr));

      // A sequence of unsafe characters will follow.
      bptr = mptr;
      if(bptr == eptr)
        break;

      // Encode one unsafe character.
      char ch = *(bptr++);
      if(ch == ' ') {
        fmt.putc('+');
        continue;
      }

      static constexpr char s_xdigits[] = "0123456789ABCDEF";
      uint32_t val = static_cast<uint8_t>(ch);

      char seq[4] = { '%' };
      seq[1] = s_xdigits[val / 16];
      seq[2] = s_xdigits[val % 16];
      fmt.putn(seq, 3);
    }
    return fmt;
  }

inline
uint32_t
do_xdigit_value(char ch)
  {
    uint32_t uch = static_cast<uint8_t>(ch);
    if(('0' <= uch) && (uch <= '9'))
      return uch - '0';

    uch |= 0x20;
    if(('a' <= uch) && (uch <= 'f'))
      return uch - 'a' + 10;

    POSEIDON_THROW("Invalid hexadecimal digit after `%`: $1", ch);
  }

cow_string&
do_decode_query(cow_string& str, const char* bptr, const char* eptr)
  {
    str.clear();

    for(auto p = bptr;  p != eptr;  ++p) {
      uint32_t uch = static_cast<uint8_t>(*p);

      if(uch == '+') {
        str += ' ';
        continue;
      }

      if(uch != '%') {
        str += static_cast<char>(uch);
        continue;
      }

      // Parse two hexadecimal digits following the percent sign.
      // Note `do_xdigit_value()` shall throw an exception if `*p` is zero
      // i.e. when `p == eptr`.
      uch = do_xdigit_value(*++p) << 4;
      uch |= do_xdigit_value(*++p);

      str += static_cast<char>(uch);
    }
    return str;
  }

} // namespace

Option_Map::
~Option_Map()
  {
  }

size_t
Option_Map::
do_bucket_index(cow_string::shallow_type key, size_t hash)
const noexcept
  {
#ifdef ROCKET_DEBUG
    ROCKET_ASSERT(this->do_key_hash(key) == hash);
#endif
    auto bptr = this->m_stor.data();
    auto eptr = bptr + this->m_stor.size();

    // Find a bucket using linear probing.
    // We keep the load factor below 1.0 so there will always be some empty buckets in
    // the table.
    auto mptr = ::rocket::get_probing_origin(bptr, eptr, hash);
    auto qbkt = ::rocket::linear_probe(bptr, mptr, mptr, eptr,
                    [&](const Bucket& r) {
                      return (r.hash == hash) &&
                             this->do_key_equal(::rocket::sref(r.key), key);
                    });
    ROCKET_ASSERT(qbkt);
    return static_cast<size_t>(qbkt - bptr);
  }

pair<const cow_string*, size_t>
Option_Map::
do_equal_range(cow_string::shallow_type key, size_t hash)
const noexcept
  {
    const auto& bkt = this->m_stor[this->do_bucket_index(key, hash)];
    switch(bkt.vstor.index()) {
      case 0:
        // If the bucket is empty, return an empty range.
        return { nullptr, 0 };

      case 1: {
        // If the bucket holds a scalar value, return it.
        const auto& s = bkt.vstor.as<1>();
        return { &s, 1 };
      }

      case 2: {
        // If the bucket holds an array of values, return its contents.
        const auto& v = bkt.vstor.as<2>();
        return { v.data(), v.size() };
      }

      default:
        ROCKET_ASSERT(false);
    }
  }

void
Option_Map::
do_reserve_more()
  {
    // Reserve more room by rehashing if the load factor would exceed 0.5.
    if(ROCKET_EXPECT(this->m_nbkt < this->m_stor.size() / 2))
      return;

    // Allocate a new table.
    ::rocket::cow_vector<Bucket> stor(this->m_nbkt * 3 | 17);
    auto bptr = stor.mut_data();
    auto eptr = bptr + stor.size();

    // Move-assign buckets into the new table.
    for(auto q = this->m_stor.mut_begin();  q != this->m_stor.end();  ++q) {
      // Find a new bucket for the key using linear probing.
      // Uniqueness has already been implied for all elements, so there is no need
      // to check for collisions.
      auto mptr = ::rocket::get_probing_origin(bptr, eptr, q->hash);
      auto qbkt = ::rocket::linear_probe(bptr, mptr, mptr, eptr,
                                         [&](const Bucket&) { return false;  });
      *qbkt = ::std::move(*q);
    }

    // Set up the new table.
    this->m_stor.swap(stor);
  }

cow_string&
Option_Map::
do_mutable_scalar(const cow_string& key, size_t hash)
  {
    this->do_reserve_more();
    auto& bkt = this->m_stor.mut(this->do_bucket_index(::rocket::sref(key), hash));
    switch(bkt.vstor.index()) {
      case 0: {
        // If the bucket is empty, construct an empty string.
        bkt.key = key;
        bkt.hash = hash;
        this->m_nbkt++;
        return bkt.vstor.emplace<1>();
      }

      case 1:
        // If the bucket holds a scalar value, return it intact.
        return bkt.vstor.as<1>();

      case 2: {
        // If the bucket holds an array of values, use the last one.
        auto v = ::std::move(bkt.vstor.as<2>());
        if(v.empty())
          return bkt.vstor.emplace<1>();

        return bkt.vstor.emplace<1>(::std::move(v.mut_back()));
      }

      default:
        ROCKET_ASSERT(false);
    }
  }

cow_string&
Option_Map::
do_mutable_append(const cow_string& key, size_t hash)
  {
    this->do_reserve_more();
    auto& bkt = this->m_stor.mut(this->do_bucket_index(::rocket::sref(key), hash));
    switch(bkt.vstor.index()) {
      case 0: {
        // If the bucket is empty, construct an empty string.
        bkt.key = key;
        bkt.hash = hash;
        this->m_nbkt++;
        return bkt.vstor.emplace<1>();
      }

      case 1: {
        // If the bucket holds a scalar value, convert it to an array, then append
        // an empty string.
        auto s = ::std::move(bkt.vstor.as<1>());

        auto& v = bkt.vstor.emplace<2>();
        v.emplace_back(::std::move(s));
        return v.emplace_back();
      }

      case 2:
        // If the bucket holds an array of values, append an empty string.
        return bkt.vstor.as<2>().emplace_back();

      default:
        ROCKET_ASSERT(false);
    }
  }

size_t
Option_Map::
do_erase(cow_string::shallow_type key, size_t hash)
  {
    size_t nerased;
    auto& bkt = this->m_stor.mut(this->do_bucket_index(key, hash));
    switch(bkt.vstor.index()) {
      case 0:
        // If the bucket is empty, there is nothing to erase.
        return 0;

      case 1:
        // If the bucket holds a scalar value, erase it.
        nerased = 1;
        bkt.vstor = nullopt;
        break;

      case 2:
        // If the bucket holds an array of values, erase its contents.
        nerased = bkt.vstor.as<2>().size();
        bkt.vstor = nullopt;
        break;

      default:
        ROCKET_ASSERT(false);
    }

    // Clear the bucket.
    bkt.key = ::rocket::sref("<empty>");
    bkt.hash = 0xDEADBEEF;
    this->m_nbkt--;

    // Reallocate buckets that follow `bkt`.
    auto bptr = this->m_stor.mut_data();
    auto eptr = bptr + this->m_stor.size();

    ::rocket::linear_probe(
      bptr,
      &bkt,
      &bkt + 1,
      eptr,
      [&](Bucket& r) {
        // Make the old bucket empty.
        auto vstor = ::std::exchange(r.vstor, nullopt);

        // Find a new bucket for the name using linear probing.
        // Uniqueness has already been implied for all elements, so there is no
        // need to check for collisions.
        auto mptr = ::rocket::get_probing_origin(bptr, eptr, r.hash);
        auto qbkt = ::rocket::linear_probe(bptr, mptr, mptr, eptr,
                                           [&](const Bucket&) { return false;  });
        ROCKET_ASSERT(qbkt);

        // Make the new bucket non-empty.
        // Note that `*qbkt` and `r` may reference the same bucket.
        qbkt->key.swap(r.key);
        qbkt->hash = r.hash;
        qbkt->vstor = ::std::move(vstor);

        // Keep probing until an empty bucket is found.
        return false;
      });

    // Return the number of elements that have been erased.
    return nerased;
  }

tinyfmt&
Option_Map::
print(tinyfmt& fmt)
const
  {
    fmt << "{\n";

    // Print all keye-value pairs.
    for(const auto& bkt : this->m_stor) {
      // Indent each element a bit.
      auto print_one = [&](const cow_string& value)
        {
          fmt << "  \"" << bkt.key << "\" = \"" << value << "\"\n";
        };

      switch(bkt.vstor.index()) {
        case 0:
          // If the bucket is empty, do nothing.
          break;

        case 1:
          // If the bucket holds a scalar value, write it after the key.
          print_one(bkt.vstor.as<1>());
          break;

        case 2:
          // If the bucket holds an array of values, write each one after the key
          // as a separated line.
          ::rocket::for_each(bkt.vstor.as<2>(), print_one);
          break;

        default:
          ROCKET_ASSERT(false);
      }
    }

    return fmt << "}\n";
  }

tinyfmt&
Option_Map::
print_url_query(tinyfmt& fmt)
const
  {
    size_t count = 0;

    // Encode all key-value pairs.
    for(const auto& bkt : this->m_stor) {
      // Spaces are encoded differently from URLs.
      auto print_one = [&](const cow_string& value)
        {
          // Separate elements with ampersands (&).
          if(count++)
            fmt << '&';

          // Spaces are encoded as plus signs (+).
          do_encode_query(fmt, bkt.key);
          fmt << '=';
          do_encode_query(fmt, value);
        };

      switch(bkt.vstor.index()) {
        case 0:
          // If the bucket is empty, do nothing.
          break;

        case 1:
          // If the bucket holds a scalar value, write it after the key.
          print_one(bkt.vstor.as<1>());
          break;

        case 2:
          // If the bucket holds an array of values, write each one after the key
          // as a separated line.
          ::rocket::for_each(bkt.vstor.as<2>(), print_one);
          break;

        default:
          ROCKET_ASSERT(false);
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

    // Ensure the string doesn't contain control characters.
    if(::rocket::any_of(str, [&](char ch) { return do_is_opt_ctype(ch, opt_ctype_control);  }))
      POSEIDON_THROW("Invalid character in URL query string `$1`", str);

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
      mptr = ::std::find_if(bptr, eptr, [&](char ch) { return ::rocket::is_any_of(ch, {'=', '&'});  });

      // Decode the key.
      cow_string key;
      do_decode_query(key, bptr, mptr);
      bptr = mptr;

      // Note a value may contain equals signs.
      if(bptr[0] == '=')
        mptr = ::std::find(++bptr, eptr, '&');

      // Decode the value.
      auto& s = this->do_mutable_append(key, this->do_key_hash(::rocket::sref(key)));
      do_decode_query(s, bptr, mptr);
      bptr = mptr;
    }
    return *this;
  }

} // namespace poseidon
