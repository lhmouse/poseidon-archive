// This file is part of Poseidon.
// Copyleft 2022, LH_Mouse. All wrongs reserved.

#include "../precompiled.ipp"
#include "socket_address.hpp"
#include "../utils.hpp"
#include <arpa/inet.h>

namespace poseidon {
namespace {

ROCKET_ALWAYS_INLINE
bool
do_match_subnet(const uint8_t* addr, size_t size,
                initializer_list<uint8_t> pattern, uint32_t bits) noexcept
  {
    ROCKET_ASSERT(pattern.size() != 0);
    ROCKET_ASSERT(bits <= size * 8);
    ROCKET_ASSERT(bits <= pattern.size() * 8);

    auto bp = addr;
    auto pp = pattern.begin();

    for(uint32_t k = 0;  k != bits / 8;  ++k)
      if(*(bp++) != *(pp++))
        return false;

    return (*bp ^ *pp) & (0xFF00 >> bits % 8);
  }

Socket_Address_Class
do_classify_ipv4(const uint8_t* addr) noexcept
  {
    // 0.0.0.0/8: Local Identification
    if(do_match_subnet(addr, 4, {0,0,0,0}, 0))
      return socket_address_class_reserved;

    // 10.0.0.0/8: Class A Private-Use
    if(do_match_subnet(addr, 4, {10,0,0,0}, 8))
      return socket_address_class_private;

    // 127.0.0.0/8: Loopback
    if(do_match_subnet(addr, 4, {127,0,0,0}, 8))
      return socket_address_class_loopback;

    // 172.16.0.0/12: Class B Private-Use
    if(do_match_subnet(addr, 4, {172,16,0,0}, 12))
      return socket_address_class_private;

    // 169.254.0.0/16: Link Local
    if(do_match_subnet(addr, 4, {169,254,0,0}, 16))
      return socket_address_class_private;

    // 192.168.0.0/16: Class C Private-Use
    if(do_match_subnet(addr, 4, {192,168,0,0}, 16))
      return socket_address_class_private;

    // 224.0.0.0/4: Class D Multicast
    if(do_match_subnet(addr, 4, {224,0,0,0}, 4))
      return socket_address_class_multicast;

    // 240.0.0.0/4: Class E
    if(do_match_subnet(addr, 4, {240,0,0,0}, 4))
      return socket_address_class_reserved;

    // The others are public IPv4 addresses.
    return socket_address_class_public;
  }

Socket_Address_Class
do_classify_ipv6(const uint8_t* addr) noexcept
  {
    // ::/128: Unspecified
    if(do_match_subnet(addr, 16, {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0}, 128))
      return socket_address_class_reserved;

    // ::1/128: Loopback
    if(do_match_subnet(addr, 16, {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1}, 128))
      return socket_address_class_loopback;

    // ::ffff:0:0/96: IPv4-mapped
    if(do_match_subnet(addr, 16, {0,0,0,0,0,0,0,0,0,0,0xFF,0xFF,0,0,0,0}, 96))
      return do_classify_ipv4(addr + 12);

    // 64:ff9b::/96: IPv4 to IPv6
    if(do_match_subnet(addr, 16, {0,0x64,0xFF,0x9B,0,0,0,0,0,0,0,0,0,0,0,0}, 96))
      return do_classify_ipv4(addr + 12);

    // 64:ff9b:1::/48: Local-Use IPv4/IPv6
    if(do_match_subnet(addr, 16, {0,0x64,0xFF,0x9B,1,0,0,0,0,0,0,0,0,0,0,0}, 48))
      return socket_address_class_private;

    // 100::/64: Discard-Only
    if(do_match_subnet(addr, 16, {0x01,0x00,0,0,0,0,0,0,0,0,0,0,0,0,0,0}, 64))
      return socket_address_class_reserved;

    // 2001:db8::/32: Documentation
    if(do_match_subnet(addr, 16, {0x20,0x01,0x0D,0xB8,0,0,0,0,0,0,0,0,0,0,0,0}, 32))
      return socket_address_class_reserved;

    // 2002::/16: 6to4
    if(do_match_subnet(addr, 16, {0x20,0x02,0,0,0,0,0,0,0,0,0,0,0,0,0,0}, 16))
      return do_classify_ipv4(addr + 2);

    // fc00::/7: Unique Local Unicast
    if(do_match_subnet(addr, 16, {0xFC,0x00,0,0,0,0,0,0,0,0,0,0,0,0,0,0}, 7))
      return socket_address_class_private;

    // fe80::/10: Link-Scoped Unicast
    if(do_match_subnet(addr, 16, {0xFE,0x80,0,0,0,0,0,0,0,0,0,0,0,0,0,0}, 10))
      return socket_address_class_private;

    // ff00::/8: Multicast
    if(do_match_subnet(addr, 16, {0xFF,0x00,0,0,0,0,0,0,0,0,0,0,0,0,0,0}, 8))
      return socket_address_class_multicast;

    // The others are public IPv6 addresses.
    return socket_address_class_public;
  }

struct cstr_buf
  {
    char sso[56];
    cow_string str;

    void
    assign(const char* bp, const char* ep)
      {
        size_t len = (size_t) (ep - bp);

        if(len + 1 <= sizeof(this->sso)) {
          ::std::memcpy(this->sso, bp, len);
          this->sso[len] = 0;
          this->str = ::rocket::sref(this->sso, len);
        }
        else
          this->str.assign(bp, ep);
      }

    const char*
    c_str() const noexcept
      { return this->str.c_str();  }
  };

}  // namespace

Socket_Address::
Socket_Address(const cow_string& str)
  {
    if(!this->parse(str))
      POSEIDON_THROW((
          "Could not parse socket address string `$1`"),
          str);
  }

Socket_Address_Class
Socket_Address::
classify() const noexcept
  {
    return do_classify_ipv6((const uint8_t*) &(this->m_addr));
  }

bool
Socket_Address::
parse(const cow_string& str)
  {
    this->clear();

    if(str.empty())
      return true;

    uint8_t* const addr = (uint8_t*) &(this->m_addr);
    const char* const bp = str.safe_c_str();
    const char* const ep = bp + str.size();
    const char* rp;

    int family;
    cstr_buf host;
    ::rocket::ascii_numget numg;
    uint64_t port = 0;

    if(*bp != '[') {
      // Parse an IPv4 address.
      family = AF_INET;

      rp = ::std::find(bp, ep, ':');
      if(*rp == ':')
        host.assign(bp, rp);
      else
        host.str = str;
    }
    else {
      // Parse an IPv6 address.
      family = AF_INET6;

      rp = ::std::find(bp, ep, ']');
      if(*rp != ']')
        return false;

      host.assign(bp + 1, rp);
      rp = ::std::find(rp + 1, ep, ':');
    }

    if(*rp == ':') {
      // Parse the port number.
      rp ++;
      if(!numg.parse_U(rp, ep))
        return false;

      if(!numg.cast_U(port, 0, 0xFFFF))
        return false;
    }

    // Check for garbage characters.
    if(rp != ep)
      return false;

    if(family == AF_INET) {
      // Parse the IPv4 address as a mapped one.
      if(::inet_pton(family, host.c_str(), addr + 12) == 0)
        return false;

      ::memset(addr, 0, 10);
      ::memset(addr + 10, 0xFF, 2);
    }
    else {
      // Parse the IPv6 address.
      if(::inet_pton(family, host.c_str(), addr) == 0)
        return false;
    }

    // Set the port number.
    this->m_port = (uint16_t) port;
    return true;
  }

tinyfmt&
Socket_Address::
print(tinyfmt& fmt) const
  {
    const uint8_t* const addr = (const uint8_t*) &(this->m_addr);
    char sbuf[56];
    const char* host = nullptr;

    if(do_match_subnet(addr, 16, {0,0,0,0,0,0,0,0,0,0,0xFF,0xFF,0,0,0,0}, 96)) {
      // Format it as an IPv4 address.
      host = ::inet_ntop(AF_INET, addr + 12, sbuf, sizeof(sbuf));
      if(!host)
        return fmt << "(invalid IPv4 address)";

      fmt << host;
    }
    else {
      // Format it as an IPv6 address.
      host = ::inet_ntop(AF_INET6, addr, sbuf, sizeof(sbuf));
      if(!host)
        return fmt << "(invalid IPv6 address)";

      fmt << '[' << host << ']';
    }

    // Format the port number.
    fmt << ':' << this->m_port;
    return fmt;
 }

cow_string
Socket_Address::
print_to_string() const
  {
    ::rocket::tinyfmt_str fmt;
    this->print(fmt);
    return fmt.extract_string();
  }

}  // namespace poseidon
