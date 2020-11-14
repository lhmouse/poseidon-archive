// This file is part of Poseidon.
// Copyleft 2020, LH_Mouse. All wrongs reserved.

#include "../precompiled.hpp"
#include "url.hpp"
#include "../util.hpp"

namespace poseidon {
namespace {

// Character categories are defined in
//   https://tools.ietf.org/html/rfc3986
// This is not a strictly compliant implementation, as tolerance is generally
// preferred to conformance.
enum : uint8_t
  {
    url_ctype_alpha       = 0x01,  // ALPHA
    url_ctype_digit       = 0x02,  // DIGIT
    url_ctype_hex_digit   = 0x04,  // HEXDIG
    url_ctype_pchar       = 0x08,  // pchar
    url_ctype_fragment    = 0x10,  // query & fragment
    url_ctype_unreserved  = 0x20,  // unreserved
    url_ctype_gen_delim   = 0x40,  // gen-delims
    url_ctype_sub_delim   = 0x80,  // sub-delims
    url_ctype_reserved    = url_ctype_gen_delim | url_ctype_sub_delim,
  };

constexpr uint8_t s_url_ctype_table[128] =
  {
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x98, 0x00, 0x40, 0x98, 0x00, 0x98, 0x98,
    0x98, 0x98, 0x98, 0x98, 0x98, 0x38, 0x38, 0x50,
    0x3E, 0x3E, 0x3E, 0x3E, 0x3E, 0x3E, 0x3E, 0x3E,
    0x3E, 0x3E, 0x58, 0x98, 0x00, 0x98, 0x00, 0x50,
    0x58, 0x3D, 0x3D, 0x3D, 0x3D, 0x3D, 0x3D, 0x39,
    0x39, 0x39, 0x39, 0x39, 0x39, 0x39, 0x39, 0x39,
    0x39, 0x39, 0x39, 0x39, 0x39, 0x39, 0x39, 0x39,
    0x39, 0x39, 0x39, 0x40, 0x00, 0x40, 0x00, 0x38,
    0x00, 0x3D, 0x3D, 0x3D, 0x3D, 0x3D, 0x3D, 0x39,
    0x39, 0x39, 0x39, 0x39, 0x39, 0x39, 0x39, 0x39,
    0x39, 0x39, 0x39, 0x39, 0x39, 0x39, 0x39, 0x39,
    0x39, 0x39, 0x39, 0x00, 0x00, 0x00, 0x38, 0x00,
  };

constexpr
uint8_t
do_get_url_ctype(char c)
  noexcept
  { return (uint8_t(c) < 128) ? s_url_ctype_table[uint8_t(c)] : 0;  }

constexpr
bool
do_is_url_ctype(char c, uint8_t mask)
  noexcept
  { return do_get_url_ctype(c) & mask;  }

tinyfmt&
do_percent_encode(tinyfmt& fmt, char ch)
  {
    static constexpr char s_xdigits[] = "0123456789ABCDEF";
    uint32_t val = static_cast<uint8_t>(ch);

    char str[4] = { '%' };
    str[1] = s_xdigits[val / 16];
    str[2] = s_xdigits[val % 16];
    return fmt.putn(str, 3);
  }

template<typename PredT>
constexpr
const char*
do_find_if_not(const char* bptr, const char* eptr, PredT&& pred)
  {
    return ::std::find_if(bptr, eptr, [&](char ch) { return !pred(ch);  });
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
do_percent_decode(cow_string& str, const char* bptr, const char* eptr)
  {
    str.clear();

    for(auto p = bptr;  p != eptr;  ++p) {
      uint32_t uch = static_cast<uint8_t>(*p);
      if(uch == '%') {
        uch = do_xdigit_value(*++p) << 4;
        uch |= do_xdigit_value(*++p);
      }
      str += static_cast<char>(uch);
    }
    return str;
  }

}  // namespace

URL::~URL()
  {
  }

URL&
URL::
set_scheme(const cow_string& val)
  {
    // The scheme string is always in lowercase.
    cow_string str = val;
    if(val.size()) {
      // The first character must be an alphabetic character.
      //   scheme = ALPHA *( ALPHA / DIGIT / "+" / "-" / "." )
      if(!do_is_url_ctype(val[0], url_ctype_alpha))
        POSEIDON_THROW("Invalid URL scheme: $1", val);

      for(size_t k = 0;  k < val.size();  ++k) {
        // Convert alphabetic characters into lowercase.
        if(do_is_url_ctype(val[k], url_ctype_alpha)) {
          str.mut(k) |= 0x20;
          continue;
        }

        // Other characters require no conversion.
        if(do_is_url_ctype(val[k], url_ctype_digit))
          continue;

        if(::rocket::is_any_of(val[k], {'+', '-', '.'}))
          continue;

        // Reject invalid characters.
        POSEIDON_THROW("Invalid URL scheme: $1", val);
      }
    }

    // Set the new scheme string.
    this->m_scheme = ::std::move(str);
    return *this;
  }

URL&
URL::
set_userinfo(const cow_string& val)
  {
    this->m_userinfo = val;
    return *this;
  }

URL&
URL::
set_host(const cow_string& val)
  {
    // Host names that are not IP addresses are all acceptable.
    // Note this check does not fail if `val` is empty.
    if(val[0] == '[') {
      // Validate the IP address.
      if(val.back() != ']')
        POSEIDON_THROW("Missing ']' after IP address: $1", val);

      if(val.size() == 2)
        POSEIDON_THROW("Empty IP address: $1", val);

      for(size_t k = 1;  k < val.size() - 1;  ++k) {
        // Hexadecimal characters are copied verbatim.
        if(do_is_url_ctype(val[k], url_ctype_hex_digit))
          continue;

        // Delimiters are allowed.
        if(::rocket::is_any_of(val[k], {'.', ':'}))
          continue;

        // Reject invalid characters.
        POSEIDON_THROW("Invalid character in IP address: $1", val);
      }
    }

    // Set the new host name.
    this->m_host = val;
    return *this;
  }

uint16_t
URL::
default_port()
  const noexcept
  {
    // Look up the well-known port for the current scheme.
    // Note the scheme string is always in lowercase.
    if(this->m_scheme == ::rocket::sref("http"))
      return 80;

    if(this->m_scheme == ::rocket::sref("https"))
      return 443;

    if(this->m_scheme == ::rocket::sref("ws"))
      return 80;

    if(this->m_scheme == ::rocket::sref("wss"))
      return 443;

    if(this->m_scheme == ::rocket::sref("ftp"))
      return 21;

    // Return zero to indicate an unknown port.
    return 0;
  }

URL&
URL::
set_port(uint16_t val)
  {
    this->m_port = val;
    return *this;
  }

URL&
URL::
set_path(const cow_string& val)
  {
    this->m_path = val;
    return *this;
  }

URL&
URL::
set_raw_query(const cow_string& val)
  {
    for(size_t k = 0;  k < val.size();  ++k) {
      // Ensure the query string does not contain unsafe characters.
      //   query = *( pchar / "/" / "?" )
      if(do_is_url_ctype(val[k], url_ctype_pchar))
        continue;

      if(::rocket::is_any_of(val[k], {'/', '?'}))
        continue;

      // Reject invalid characters.
      POSEIDON_THROW("Invalid character in query string: $1", val);
    }

    // Set the new query string.
    this->m_raw_query = val;
    return *this;
  }

URL&
URL::
set_raw_fragment(const cow_string& val)
  {
    for(size_t k = 0;  k < val.size();  ++k) {
      // Ensure the fragment string does not contain unsafe characters.
      //   fragment = *( pchar / "/" / "?" )
      if(do_is_url_ctype(val[k], url_ctype_pchar))
        continue;

      if(::rocket::is_any_of(val[k], {'/', '?'}))
        continue;

      // Reject invalid characters.
      POSEIDON_THROW("Invalid character in fragment string: $1", val);
    }

    // Set the new fragment string.
    this->m_raw_fragment = val;
    return *this;
  }

URL&
URL::
clear()
  noexcept
  {
    this->m_scheme.clear();
    this->m_userinfo.clear();
    this->m_host.clear();
    this->m_port.reset();
    this->m_path.clear();
    this->m_raw_query.clear();
    this->m_raw_fragment.clear();
    return *this;
  }

URL&
URL::
swap(URL& other)
  noexcept
  {
    this->m_scheme.swap(other.m_scheme);
    this->m_userinfo.swap(other.m_userinfo);
    this->m_host.swap(other.m_host);
    this->m_port.swap(other.m_port);
    this->m_path.swap(other.m_path);
    this->m_raw_query.swap(other.m_raw_query);
    this->m_raw_fragment.swap(other.m_raw_fragment);
    return *this;
  }

tinyfmt&
URL::
print(tinyfmt& fmt)
  const
  {
    // If a scheme field is present, write it.
    if(this->m_scheme.size())
      fmt << this->m_scheme << "://";

    // If a host name field is present, write it.
    // The userinfo and port fields are ignored without a host name.
    if(this->m_host.size()) {
      // If a userinfo field is present, write it.
      if(this->m_userinfo.size()) {
        // Escape unsafe characters.
        //   userinfo = *( unreserved / pct-encoded / sub-delims / ":" )
        ::rocket::for_each(this->m_userinfo,
            [&](char ch) {
              if(do_is_url_ctype(ch, url_ctype_unreserved | url_ctype_sub_delim))
                fmt << ch;
              else if(ch == ':')
                fmt << ch;
              else
                do_percent_encode(fmt, ch);
            });

        fmt << '@';
      }

      if((this->m_host[0] == '[') && (this->m_host.back() == ']')) {
        // If the host name is an IP address in brackets, write it verbatim.
        //   IP-literal = "[" ( IPv6address / IPvFuture  ) "]"
        //   IPvFuture = "v" 1*HEXDIG "." 1*( unreserved / sub-delims / ":" )
        fmt << this->m_host;
      }
      else {
        // Otherwise, it is treated as a reg-name.
        //   reg-name = *( unreserved / pct-encoded / sub-delims )
        ::rocket::for_each(this->m_host,
            [&](char ch) {
              if(do_is_url_ctype(ch, url_ctype_unreserved | url_ctype_sub_delim))
                fmt << ch;
              else
                do_percent_encode(fmt, ch);
            });
      }

      // If a port field is present, write it.
      if(this->m_port)
        fmt << ':' << *(this->m_port);
    }

    // Write the path.
    fmt << '/';

    ::rocket::for_each(this->m_path,
        [&](char ch) {
          if(do_is_url_ctype(ch, url_ctype_pchar))
            fmt << ch;
          else if(ch == '/')
            fmt << ch;
          else
            do_percent_encode(fmt, ch);
        });

    // If a query string field is present, write it.
    if(this->m_raw_query.size())
      fmt << '?' << this->m_raw_query;

    // If a fragment field is present, write it.
    // This has to be the last field.
    if(this->m_raw_fragment.size())
      fmt << '#' << this->m_raw_fragment;

    return fmt;
  }

URL&
URL::
parse(const cow_string& str)
  {
    // Destroy existent data.
    // If this function fails, the contents of `*this` are undefined.
    this->clear();

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

    // Check for a scheme first.
    // If no scheme string can be accepted, `bptr` shall be left intact.
    // The URL may start with a scheme, userinfo or host name field.
    // A leading alphabetic character may initiate a scheme or host name.
    if(do_is_url_ctype(bptr[0], url_ctype_alpha)) {
      mptr = do_find_if_not(bptr + 1, eptr,
                 [&](char ch) {
                   return do_is_url_ctype(ch, url_ctype_alpha | url_ctype_digit);
                 });

      if((mptr[0] == ':') && (mptr[1] == '/') && (mptr[2] == '/')) {
        // Accept the scheme.
        this->m_scheme = ascii_lowercase(cow_string(bptr, mptr));
        bptr = mptr + 3;
      }
    }

    // Check for a userinfo.
    mptr = do_find_if_not(bptr, eptr,
                 [&](char ch) {
                   return do_is_url_ctype(ch, url_ctype_unreserved | url_ctype_sub_delim) ||
                          ::rocket::is_any_of(ch, {'%', ':'});
                 });

    if(mptr[0] == '@') {
      // Accept the userinfo.
      do_percent_decode(this->m_userinfo, bptr, mptr);
      bptr = mptr + 1;
    }

    // Check for a host name.
    // The host name may be an IP address in a pair of bracket. Colons are
    // allowed inside brackets, but not outside.
    size_t brackets = bptr[0] == '[';
    mptr = do_find_if_not(bptr + brackets, eptr,
                 [&](char ch) {
                   return do_is_url_ctype(ch, url_ctype_unreserved | url_ctype_sub_delim) ||
                          (ch == (brackets ? ':' : '%'));
                 });

    if(brackets) {
      // Check the IP address.
      if(*mptr != ']')
        POSEIDON_THROW("Missing ']' after IP address: $1", str);

      if(mptr - bptr == 1)
        POSEIDON_THROW("Empty IP address: $1", str);

      mptr += 1;
    }

    if(bptr != mptr) {
      // Accept the host name.
      do_percent_decode(this->m_host, bptr, mptr);
      bptr = mptr;

      // Check for a port number.
      if(mptr[0] == ':') {
        mptr = do_find_if_not(bptr + 1, eptr,
                 [&](char ch) {
                   return do_is_url_ctype(ch, url_ctype_digit);
                 });

        if(mptr - bptr == 1)
          POSEIDON_THROW("Missing port number after `:`: $1", str);

        ::rocket::ascii_numget numg;
        if(!numg.parse_U(++bptr, mptr, 10))
          POSEIDON_THROW("Invalid port number: $1", str);

        if(bptr != mptr)
          POSEIDON_THROW("Port number out of range: $1", str);

        uint64_t val;
        if(!numg.cast_U(val, 0, 65535))
          POSEIDON_THROW("Port number out of range: $1", str);

        this->m_port = static_cast<uint16_t>(val);
      }
    }

    // Check for a path.
    if(bptr[0] == '/') {
      mptr = do_find_if_not(bptr + 1, eptr,
                 [&](char ch) {
                   return do_is_url_ctype(ch, url_ctype_pchar) ||
                          ::rocket::is_any_of(ch, {'%', '/'});
                 });

      // Accept the path without the leading slash.
      do_percent_decode(this->m_path, bptr + 1, mptr);
      bptr = mptr;
    }

    // Check for a query string.
    if(bptr[0] == '?') {
      mptr = do_find_if_not(bptr + 1, eptr,
                 [&](char ch) {
                   return do_is_url_ctype(ch, url_ctype_pchar) ||
                          ::rocket::is_any_of(ch, {'/', '?'});
                 });

      // Accept the query string without the question mark.
      // Note that it is not decoded.
      this->m_raw_query.assign(bptr + 1, mptr);
      bptr = mptr;
    }

    // Check for a fragment.
    if(bptr[0] == '#') {
      mptr = do_find_if_not(bptr + 1, eptr,
                 [&](char ch) {
                   return do_is_url_ctype(ch, url_ctype_pchar) ||
                          ::rocket::is_any_of(ch, {'/', '?'});
                 });

      // Accept the query string without the hashtag.
      do_percent_decode(this->m_raw_fragment, bptr + 1, mptr);
      bptr = mptr;
    }

    // All characters shall have been consumed so far.
    if(bptr != eptr)
      POSEIDON_THROW("Invalid URL string: $1", str);

    return *this;
  }

}  // namespace poseidon
