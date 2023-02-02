// This file is part of Poseidon.
// Copyleft 2022, LH_Mouse. All wrongs reserved.

#include "../precompiled.ipp"
#include "socket_address.hpp"
#include "../utils.hpp"
#include <arpa/inet.h>
#include <http_parser.h>

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

    if((*bp ^ *pp) & (0xFF00 >> bits % 8))
      return false;

    return true;
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

    // Break down the host:port string as a URL.
    ::http_parser_url url = { };
    if(::http_parser_parse_url(str.data(), str.size(), true, &url) != 0)
      return false;

    if(url.field_set != (1U << UF_HOST | 1U << UF_PORT))
      return false;

    // Parse the host string.
    const char* host = str.data() + url.field_data[UF_HOST].off;
    size_t hostlen = url.field_data[UF_HOST].len;
    char sbuf[64];
    int family = AF_INET6;
    uint8_t* addr = (uint8_t*) &(this->m_addr);

    if((hostlen < 1) || (hostlen > 63))
      return false;

    if(host[hostlen] != ']') {
      family = AF_INET;
      ::memset(addr, 0x00, 10);
      addr += 10;
      ::memset(addr, 0xFF, 2);
      addr += 2;
    }

    ::memcpy(sbuf, host, hostlen);
    sbuf[hostlen] = 0;
    host = sbuf;

    if(::inet_pton(family, host, addr) == 0)
      return false;

    this->m_port = url.port;
    return true;
  }

tinyfmt&
Socket_Address::
print(tinyfmt& fmt) const
  {
    int family = AF_INET6;
    const uint8_t* addr = (const uint8_t*) &(this->m_addr);
    char sbuf[64];
    char* host;

    if(do_match_subnet(addr, 16, {0,0,0,0,0,0,0,0,0,0,0xFF,0xFF,0,0,0,0}, 96)) {
      family = AF_INET;
      addr += 12;
    }

    host = (char*) ::inet_ntop(family, addr, sbuf + 1, sizeof(sbuf) - 1);
    if(!host)
      return fmt << "(invalid IP address)";

    if(family != AF_INET) {
      host --;
      host[0] = '[';
      ::strcat(host, "]:");
    }
    else
      ::strcat(host, ":");

    fmt << host << this->m_port;
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
