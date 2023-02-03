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
do_match_subnet(const void* addr, const void* mask, uint32_t bits) noexcept
  {
    return (::memcmp(addr, mask, bits / 8) == 0) &&
           (((*((const uint8_t*) addr + bits / 8) ^ *((const uint8_t*) mask + bits / 8))
             & (0xFF00U >> bits % 8)) == 0);
  }

inline
IP_Address_Class
do_classify_ipv4_generic(const void* addr) noexcept
  {
    // 0.0.0.0/32: Unspecified
    if(do_match_subnet(addr, "\x00\x00\x00\x00", 32))
      return ip_address_class_unspecified;

    // 0.0.0.0/8: Local Identification
    if(do_match_subnet(addr, "\x00", 8))
      return ip_address_class_reserved;

    // 10.0.0.0/8: Class A Private-Use
    if(do_match_subnet(addr, "\x0A", 8))
      return ip_address_class_private;

    // 127.0.0.0/8: Loopback
    if(do_match_subnet(addr, "\x7F", 8))
      return ip_address_class_loopback;

    // 172.16.0.0/12: Class B Private-Use
    if(do_match_subnet(addr, "\xAC\x10", 12))
      return ip_address_class_private;

    // 169.254.0.0/16: Link Local
    if(do_match_subnet(addr, "\xA9\xFE", 16))
      return ip_address_class_link_local;

    // 192.168.0.0/16: Class C Private-Use
    if(do_match_subnet(addr, "\xC0\xA8", 16))
      return ip_address_class_private;

    // 224.0.0.0/4: Class D Multicast
    if(do_match_subnet(addr, "\xE0", 4))
      return ip_address_class_multicast;

    // 240.0.0.0/4: Class E
    if(do_match_subnet(addr, "\xF0", 4))
      return ip_address_class_reserved;

    // Default
    return ip_address_class_public;
  }

inline
IP_Address_Class
do_classify_ipv6_generic(const void* addr) noexcept
  {
    // ::ffff:0:0/96: IPv4-mapped
    if(do_match_subnet(addr, "\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xFF\xFF", 96))
      return do_classify_ipv4_generic((const uint8_t*) addr + 12);

    // ::/128: Unspecified
    if(do_match_subnet(addr, "\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00", 128))
      return ip_address_class_unspecified;

    // ::1/128: Loopback
    if(do_match_subnet(addr, "\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x01", 128))
      return ip_address_class_loopback;

    // 64:ff9b::/96: IPv4 to IPv6
    if(do_match_subnet(addr, "\x00\x64\xFF\x9B\x00\x00\x00\x00\x00\x00\x00\x00", 96))
      return do_classify_ipv4_generic((const uint8_t*) addr + 12);

    // 64:ff9b:1::/48: Local-Use IPv4/IPv6
    if(do_match_subnet(addr, "\x00\x64\xFF\x9B\x00\x01", 48))
      return do_classify_ipv4_generic((const uint8_t*) addr + 12);

    // 100::/64: Discard-Only
    if(do_match_subnet(addr, "\x01\x00\x00\x00\x00\x00\x00\x00", 64))
      return ip_address_class_reserved;

    // 2001:db8::/32: Documentation
    if(do_match_subnet(addr, "\x20\x01\x0D\xB8", 32))
      return ip_address_class_reserved;

    // 2002::/16: 6to4
    if(do_match_subnet(addr, "\x20\x02", 16))
      return do_classify_ipv4_generic((const uint8_t*) addr + 2);

    // fc00::/7: Unique Local Unicast
    if(do_match_subnet(addr, "\xFC\x00", 7))
      return ip_address_class_private;

    // fe80::/10: Link-Scoped Unicast
    if(do_match_subnet(addr, "\xFE\x80", 10))
      return ip_address_class_link_local;

    // ff00::/8: Multicast
    if(do_match_subnet(addr, "\xFF\x00", 8))
      return ip_address_class_multicast;

    // Default
    return ip_address_class_public;
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

IP_Address_Class
Socket_Address::
classify() const noexcept
  {
    return do_classify_ipv6_generic(&(this->m_addr));
  }

bool
Socket_Address::
parse(const cow_string& str)
  {
    this->clear();

    if(str.empty())
      return true;

    if(str.size() >= UINT16_MAX)
      return false;

    // Break down the host:port string as a URL.
    ::http_parser_url url;
    url.field_set = 0;
    url.port = 0;

    if(::http_parser_parse_url(str.data(), str.size(), true, &url) != 0)
      return false;

    if(url.field_set != (1U << UF_HOST | 1U << UF_PORT))
      return false;

    const char* host = str.data() + url.field_data[UF_HOST].off;
    size_t hostlen = url.field_data[UF_HOST].len;
    uint8_t* addr = (uint8_t*) &(this->m_addr);
    int family = AF_INET6;
    char sbuf[64];

    if((hostlen < 1) || (hostlen > 63))
      return false;

    if(host[hostlen] != ']') {
      // IPv4
      ::memcpy(addr, "\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xFF\xFF", 12);
      addr += 12;
      family = AF_INET;
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
    const uint8_t* addr = (const uint8_t*) &(this->m_addr);
    int family = AF_INET6;
    char sbuf[64];
    char* host;

    if(do_match_subnet(addr, "\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xFF\xFF", 96)) {
      // IPv4
      addr += 12;
      family = AF_INET;
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
