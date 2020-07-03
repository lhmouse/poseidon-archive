// This file is part of Poseidon.
// Copyleft 2020, LH_Mouse. All wrongs reserved.

#include "../precompiled.hpp"
#include "url.hpp"
#include "../utilities.hpp"

namespace poseidon {
namespace {

// Character categories are defined in
//   https://tools.ietf.org/html/rfc3986
// This is not a strictly compliant implementation, as tolerance is generally
// preferred to conformance.
enum : uint8_t
  {
    cctype_alpha       = 0x01,  // ALPHA
    cctype_digit       = 0x02,  // DIGIT
    cctype_hex_digit   = 0x04,  // HEXDIG
    cctype_pchar       = 0x08,  // pchar
    cctype_fragment    = 0x10,  // query & fragment
    cctype_unreserved  = 0x20,  // unreserved
    cctype_gen_delim   = 0x40,  // gen-delims
    cctype_sub_delim   = 0x80,  // sub-delims
    cctype_reserved    = cctype_gen_delim | cctype_sub_delim,
  };

constexpr uint8_t s_cctype_table[128] =
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
do_get_cctype(char c)
noexcept
  { return (uint8_t(c) < 128) ? s_cctype_table[uint8_t(c)] : 0;  }

constexpr
bool
do_is_cctype(char c, uint8_t mask)
noexcept
  { return do_get_cctype(c) & mask;  }

class Percent_Encode
  {
  private:
    char m_ch;

  public:
    constexpr
    Percent_Encode(char ch)
    noexcept
      : m_ch(ch)
      { }

  public:
    constexpr
    uintptr_t
    value()
    const noexcept
      { return static_cast<uint8_t>(this->m_ch);  }
  };

constexpr
Percent_Encode
do_percent_encode(char ch)
noexcept
  { return Percent_Encode(ch);  }

inline
tinyfmt&
operator<<(tinyfmt& fmt, const Percent_Encode& pctec)
  {
    static constexpr char s_xdigits[] = "0123456789ABCDEF";
    uintptr_t val = pctec.value();

    char str[4] = { '%' };
    str[1] = s_xdigits[val / 16];
    str[2] = s_xdigits[val % 16];
    return fmt.putn(str, 3);
  }

cow_string&
do_convert_to_lowercase(cow_string& str, const char* bptr, const char* eptr)
  {
    str.clear();

    for(auto p = bptr;  p != eptr;  ++p) {
      char32_t uch = *p & 0xFF;
      if(do_is_cctype(*p, cctype_alpha)) {
        uch |= 0x20;
      }
      str += static_cast<char>(uch);
    }
    return str;
  }

template<typename PredT>
constexpr
const char*
do_find_if_not(const char* bptr, const char* eptr, PredT&& pred)
  {
    return ::std::find_if(bptr, eptr, [&](char ch) { return !pred(ch);  });
  }

inline
char32_t
do_xdigit_value(char ch)
  {
    char32_t uch = ch & 0xFF;
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
      char32_t uch = *p & 0xFF;
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

uint16_t
URL::
do_get_default_port()
const noexcept
  {
    // Look up the well-known port for the current scheme.
    // Note the scheme string is always in lowercase.
    if(this->m_scheme == "http")
      return 80;

    if(this->m_scheme == "https")
      return 443;

    if(this->m_scheme == "ws")
      return 80;

    if(this->m_scheme == "wss")
      return 443;

    if(this->m_scheme == "ftp")
      return 21;

    // Return zero to indicate an unknown port.
    return 0;
  }

cow_string&
URL::
do_verify_and_set_scheme(cow_string&& val)
  {
    // An empty scheme is valid.
    if(val.empty())
      return this->m_scheme.clear();

    // The first character must be an alphabetic character.
    //   scheme = ALPHA *( ALPHA / DIGIT / "+" / "-" / "." )
    if(!do_is_cctype(val[0], cctype_alpha))
      POSEIDON_THROW("Invalid URL scheme: $1", val);

    for(size_t k = 0;  k < val.size();  ++k) {
      // Convert alphabetic characters into lowercase.
      if(do_is_cctype(val[k], cctype_alpha)) {
        val.mut(k) |= 0x20;
        continue;
      }

      // Other characters require no conversion.
      if(do_is_cctype(val[k], cctype_digit))
        continue;

      if(::rocket::is_any_of(val[k], {'+','-','.'}))
        continue;

      // Reject invalid characters.
      POSEIDON_THROW("Invalid URL scheme: $1", val);
    }

    // Set the new scheme string.
    return this->m_scheme = ::std::move(val);
  }

cow_string&
URL::
do_verify_and_set_host(cow_string&& val)
  {
    // Host names that are not IP addresses are all acceptable.
    // Note this check does not fail if `val` is empty.
    if(val[0] != '[')
      return this->m_host = ::std::move(val);

    if(val.back() != ']')
      POSEIDON_THROW("Missing ']' after IP address: $1", val);

    if(val.size() == 2)
      POSEIDON_THROW("Empty IP address: $1", val);

    for(size_t k = 1;  k < val.size() - 1;  ++k) {
      // Convert alphabetic characters into lowercase.
      if(do_is_cctype(val[k], cctype_alpha)) {
        val.mut(k) |= 0x20;
        continue;
      }

      if(do_is_cctype(val[k], cctype_digit))
        continue;

      if(::rocket::is_any_of(val[k], {'.',':'}))
        continue;

      // Reject invalid characters.
      POSEIDON_THROW("Invalid character in IP address: $1", val);
    }

    // Set the new host name.
    return this->m_host = ::std::move(val);
  }

cow_string&
URL::
do_verify_and_set_query(cow_string&& val)
  {
    for(size_t k = 0;  k < val.size();  ++k) {
      // Ensure the query string does not contain unsafe characters.
      //   query = *( pchar / "/" / "?" )
      if(do_is_cctype(val[k], cctype_pchar))
        continue;

      if(::rocket::is_any_of(val[k], {'/','?'}))
        continue;

      // Reject invalid characters.
      POSEIDON_THROW("Invalid character in query string: $1", val);
    }

    // Set the new query string.
    return this->m_query = ::std::move(val);
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
            [&](char ch) -> tinyfmt& {
              if(do_is_cctype(ch, cctype_unreserved | cctype_sub_delim))
                return fmt << ch;
              else if(ch == ':')
                return fmt << ch;
              else
                return fmt << do_percent_encode(ch);
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
            [&](char ch) -> tinyfmt& {
              if(do_is_cctype(ch, cctype_unreserved | cctype_sub_delim))
                return fmt << ch;
              else
                return fmt << do_percent_encode(ch);
            });
      }

      // If a port field is present, write it.
      if(this->m_port)
        fmt << ':' << *(this->m_port);
    }

    // Write the path.
    fmt << '/';

    ::rocket::for_each(this->m_path,
        [&](char ch) -> tinyfmt& {
          if(do_is_cctype(ch, cctype_pchar))
            return fmt << ch;
          else if(ch == '/')
            return fmt << ch;
          else
            return fmt << do_percent_encode(ch);
        });

    // If a query string field is present, write it.
    if(this->m_query.size())
      fmt << '?' << this->m_query;

    // If a fragment field is present, write it.
    // This has to be the last field.
    if(this->m_fragment.size()) {
      fmt << '#';

      // Escape unsafe characters.
      //   fragment = *( pchar / "/" / "?" )
      ::rocket::for_each(this->m_fragment,
          [&](char ch) -> tinyfmt& {
            if(do_is_cctype(ch, cctype_pchar))
              return fmt << ch;
            else if(::rocket::is_any_of(ch, {'/','?'}))
              return fmt << ch;
            else
              return fmt << do_percent_encode(ch);
          });
    }
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
    if(do_is_cctype(bptr[0], cctype_alpha)) {
      mptr = do_find_if_not(bptr + 1, eptr,
                 [&](char ch) {
                   return do_is_cctype(ch, cctype_alpha | cctype_digit);
                 });

      if((mptr[0] == ':') && (mptr[1] == '/') && (mptr[2] == '/')) {
        // Accept the scheme.
        do_convert_to_lowercase(this->m_scheme, bptr, mptr);
        bptr = mptr + 3;
      }
    }

    // Check for a userinfo.
    mptr = do_find_if_not(bptr, eptr,
                 [&](char ch) {
                   return do_is_cctype(ch, cctype_unreserved | cctype_sub_delim) ||
                          ::rocket::is_any_of(ch, {'%',':'});
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
                   return do_is_cctype(ch, cctype_unreserved | cctype_sub_delim) ||
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
                   return do_is_cctype(ch, cctype_digit);
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
                   return do_is_cctype(ch, cctype_pchar) ||
                          ::rocket::is_any_of(ch, {'%','/'});
                 });

      // Accept the path without the leading slash.
      do_percent_decode(this->m_path, bptr + 1, mptr);
      bptr = mptr;
    }

    // Check for a query string.
    if(bptr[0] == '?') {
      mptr = do_find_if_not(bptr + 1, eptr,
                 [&](char ch) {
                   return do_is_cctype(ch, cctype_pchar) ||
                          ::rocket::is_any_of(ch, {'/','?'});
                 });

      // Accept the query string without the question mark.
      // Note that it is not decoded.
      this->m_query.assign(bptr + 1, mptr);
      bptr = mptr;
    }

    // Check for a fragment.
    if(bptr[0] == '#') {
      mptr = do_find_if_not(bptr + 1, eptr,
                 [&](char ch) {
                   return do_is_cctype(ch, cctype_pchar) ||
                          ::rocket::is_any_of(ch, {'/','?'});
                 });

      // Accept the query string without the hashtag.
      do_percent_decode(this->m_fragment, bptr + 1, mptr);
      bptr = mptr;
    }

    if(bptr != eptr)
      POSEIDON_THROW("Junk data after URL: $1", str);

    return *this;
  }

}  // namespace poseidon