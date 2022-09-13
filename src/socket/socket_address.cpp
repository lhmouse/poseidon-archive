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
do_match_subnet(const void* addr, size_t size, initializer_list<uint8_t> pattern, uint32_t bits) noexcept
  {
    ROCKET_ASSERT(pattern.size() != 0);
    ROCKET_ASSERT(bits <= size * 8);
    ROCKET_ASSERT(bits <= pattern.size() * 8);

    auto bp = (const uint8_t*) addr;
    auto pp = pattern.begin();

    for(size_t k = 0;  k != bits / 8;  ++k)
      if(*(bp++) != *(pp++))
        return false;

    return (*bp ^ *pp) & (0xFF00 >> bits % 8);
  }

Socket_Address_Class
do_classify_ipv4(const ::in_addr* addr) noexcept
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
do_classify_ipv6(const ::in6_addr* addr) noexcept
  {
    // ::/128: Unspecified
    if(do_match_subnet(addr, 16, {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0}, 128))
      return socket_address_class_reserved;

    // ::1/128: Loopback
    if(do_match_subnet(addr, 16, {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1}, 128))
      return socket_address_class_loopback;

    // ::ffff:0:0/96: IPv4-mapped
    if(do_match_subnet(addr, 16, {0,0,0,0,0,0,0,0,0,0,0xFF,0xFF,0,0,0,0}, 96))
      return do_classify_ipv4((const ::in_addr*) ((const char*) addr + 12));

    // 64:ff9b::/96: IPv4 to IPv6
    if(do_match_subnet(addr, 16, {0,0x64,0xFF,0x9B,0,0,0,0,0,0,0,0,0,0,0,0}, 96))
      return do_classify_ipv4((const ::in_addr*) ((const char*) addr + 12));

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
      return do_classify_ipv4((const ::in_addr*) ((const char*) addr + 2));

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
Socket_Address(const cow_string& host, uint16_t port)
  {
    if(!this->parse(host, port))
      POSEIDON_THROW((
          "Could not parse IP address string `$1`"),
          host);
  }

Socket_Address_Class
Socket_Address::
classify() const noexcept
  {
    if(this->m_family == AF_INET) {
      // IPv4
      return do_classify_ipv4(&(this->m_addr4.sin_addr));
    }

    if(this->m_family == AF_INET6) {
      // IPv6
      return do_classify_ipv6(&(this->m_addr6.sin6_addr));
    }

    return socket_address_class_unknown;
  }

tinyfmt&
Socket_Address::
print(tinyfmt& fmt) const
  {
    if(this->m_family == AF_INET) {
      // IPv4
      char sbuf[128];
      const char* host = ::inet_ntop(AF_INET, &(this->m_addr4.sin_addr), sbuf, sizeof(sbuf));
      if(!host)
        return fmt << "(invalid IPv4 address)";

      // Write the host and port, separated by a colon.
      return fmt << host << ":" << (uint32_t) be16toh(this->m_addr4.sin_port);
    }

    if(this->m_family == AF_INET6) {
      // IPv6
      char sbuf[128];
      const char* host = ::inet_ntop(AF_INET6, &(this->m_addr6.sin6_addr), sbuf, sizeof(sbuf));
      if(!host)
        return fmt << "(invalid IPv6 address)";

      // Write the host in brackets, followed by the port, separated by a colon.
      return fmt << "[" << host << "]:" << (uint32_t) be16toh(this->m_addr6.sin6_port);
    }

    return fmt << "(unknown address family " << (uint32_t) this->m_family << ")";
  }

cow_string
Socket_Address::
print_to_string() const
  {
    ::rocket::tinyfmt_str fmt;
    this->print(fmt);
    return fmt.extract_string();
  }

bool
Socket_Address::
parse(const cow_string& host, uint16_t port)
  {
    this->m_family = AF_UNSPEC;

    // An empty string denotes the null address.
    if(ROCKET_UNEXPECT(host.empty()))
      return true;

    if((host.front() >= '0') && (host.front() <= '9')) {
      // Assume IPv4.
      if(::inet_pton(AF_INET, host.safe_c_str(), &(this->m_addr4.sin_addr)) != 1)
        return false;

      // Set up data.
      this->m_family = AF_INET;
      this->m_addr4.sin_port = htobe16(port);
      this->m_size = sizeof(::sockaddr_in);
      return true;
    }

    if((host.front() == '[') && (host.back() == ']')) {
      // Unbracket the host string.
      cow_string ub_host;
      size_t len = host.size() - 2;
      char sbuf[128];

      if(len < sizeof(sbuf)) {
        // Use the static buffer.
        ::memcpy(sbuf, host.data() + 1, len);
        sbuf[len] = 0;
        ub_host = ::rocket::sref(sbuf, len);
      }
      else
        ub_host.assign(host.begin() + 1, host.end() - 1);

      // Try parsing it as IPv6.
      if(::inet_pton(AF_INET6, ub_host.c_str(), &(this->m_addr6.sin6_addr)) != 1)
        return false;

      // Set up data.
      this->m_family = AF_INET6;
      this->m_addr6.sin6_port = htobe16(port);
      this->m_addr6.sin6_flowinfo = 0;
      this->m_addr6.sin6_scope_id = 0;
      this->m_size = sizeof(::sockaddr_in6);
      return true;
    }

    return false;
  }

}  // namespace poseidon
