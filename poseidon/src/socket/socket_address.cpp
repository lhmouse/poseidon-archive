// This file is part of Poseidon.
// Copyleft 2020, LH_Mouse. All wrongs reserved.

#include "../precompiled.hpp"
#include "socket_address.hpp"
#include "../utils.hpp"
#include <arpa/inet.h>

namespace poseidon {
namespace {

inline ::std::bitset<32>
do_cast_ipv4(const ::in_addr& in4) noexcept
  {
    ::std::bitset<32> bits;
    bits = be32toh(in4.s_addr);
    return bits;
  }

inline ::std::bitset<32>
do_make_ipv4(const array<uint8_t, 4>& segs) noexcept
  {
    ::std::bitset<32> bits;
    for(size_t k = 0;  k != 4;  ++k)
      bits |= ::std::bitset<32>(segs[k]) << (24 - k * 8);
    return bits;
  }

inline ::std::bitset<128>
do_cast_ipv6(const ::in6_addr& in6) noexcept
  {
    ::std::bitset<128> bits;
    for(size_t k = 0;  k != 16;  ++k)
      bits |= ::std::bitset<128>(in6.s6_addr[k]) << (120 - k * 8);
    return bits;
  }

inline ::std::bitset<128>
do_make_ipv6(const array<uint16_t, 8>& segs) noexcept
  {
    ::std::bitset<128> bits;
    for(size_t k = 0;  k != 8;  ++k)
      bits |= ::std::bitset<128>(segs[k]) << (112 - k * 16);
    return bits;
  }

template<size_t N>
constexpr bool
do_match(const ::std::bitset<N>& addr, const ::std::bitset<N>& comp, size_t bits) noexcept
  {
    ::std::bitset<N> mask;
    if(bits < mask.size())
      mask.set() >>= bits;  // shift in zeroes
    mask.flip();

    ROCKET_ASSERT((comp & mask) == comp);
    return (addr & mask) == comp;
  }

Socket_Address_Class
do_classify_ipv4(const ::std::bitset<32>& addr) noexcept
  {
    // 0.0.0.0/8: Local Identification
    if(do_match(addr, do_make_ipv4({0,0,0,0}), 8))
      return socket_address_class_reserved;

    // 10.0.0.0/8: Class A Private-Use
    if(do_match(addr, do_make_ipv4({10,0,0,0}), 8))
      return socket_address_class_private;

    // 127.0.0.0/8: Loopback
    if(do_match(addr, do_make_ipv4({127,0,0,0}), 8))
      return socket_address_class_loopback;

    // 172.16.0.0/12: Class B Private-Use
    if(do_match(addr, do_make_ipv4({172,16,0,0}), 12))
      return socket_address_class_private;

    // 169.254.0.0/16: Link Local
    if(do_match(addr, do_make_ipv4({169,254,0,0}), 16))
      return socket_address_class_private;

    // 192.168.0.0/16: Class C Private-Use
    if(do_match(addr, do_make_ipv4({192,168,0,0}), 16))
      return socket_address_class_private;

    // 224.0.0.0/4: Class D Multicast
    if(do_match(addr, do_make_ipv4({224,0,0,0}), 4))
      return socket_address_class_multicast;

    // 240.0.0.0/4: Class E
    if(do_match(addr, do_make_ipv4({240,0,0,0}), 4))
      return socket_address_class_reserved;

    return socket_address_class_public;
  }

Socket_Address_Class
do_classify_ipv6(const ::std::bitset<128>& addr) noexcept
  {
    // ::/128: Unspecified
    if(do_match(addr, do_make_ipv6({0,0,0,0,0,0,0,0}), 128))
      return socket_address_class_reserved;

    // ::1/128: Loopback
    if(do_match(addr, do_make_ipv6({0,0,0,0,0,0,0,1}), 128))
      return socket_address_class_loopback;

    // ::ffff:0:0/96: IPv4-mapped
    if(do_match(addr, do_make_ipv6({0,0,0,0,0,0xffff,0,0}), 96))
      return do_classify_ipv4(addr.to_ulong());

    // 64:ff9b::/96: IPv4 to IPv6
    if(do_match(addr, do_make_ipv6({0x64,0xff9b,0,0,0,0,0,0}), 96))
      return do_classify_ipv4(addr.to_ulong());

    // 64:ff9b:1::/48: Local-Use IPv4/IPv6
    if(do_match(addr, do_make_ipv6({0x64,0xff9b,1,0,0,0,0,0}), 48))
      return socket_address_class_private;

    // 100::/64: Discard-Only
    if(do_match(addr, do_make_ipv6({0x100,0,0,0,0,0,0,0}), 64))
      return socket_address_class_reserved;

    // 2001:db8::/32: Documentation
    if(do_match(addr, do_make_ipv6({0x2001,0xdb8,0,0,0,0,0,0}), 32))
      return socket_address_class_reserved;

    // 2002::/16: 6to4
    if(do_match(addr, do_make_ipv6({0x2002,0,0,0,0,0,0,0}), 16))
      return do_classify_ipv4((addr >> 80).to_ulong());

    // 4000::/2: Reserved
    if(do_match(addr, do_make_ipv6({0x4000,0,0,0,0,0,0,0}), 2))
      return socket_address_class_reserved;

    // FC00::/7: Unique Local Unicast
    if(do_match(addr, do_make_ipv6({0xFC00,0,0,0,0,0,0,0}), 7))
      return socket_address_class_private;

    // FE80::/10: Link-Scoped Unicast
    if(do_match(addr, do_make_ipv6({0xFE80,0,0,0,0,0,0,0}), 10))
      return socket_address_class_private;

    // FF00::/8: Multicast
    if(do_match(addr, do_make_ipv6({0xFF00,0,0,0,0,0,0,0}), 8))
      return socket_address_class_multicast;

    // C000::/2: Reserved (other than Link-Scoped Unicast and Multicast)
    if(do_match(addr, do_make_ipv6({0xC000,0,0,0,0,0,0,0}), 2))
      return socket_address_class_reserved;

    return socket_address_class_public;
  }

}  // namespace

static_assert(::std::is_trivially_copyable<Socket_Address>::value &&
              ::std::is_trivially_destructible<Socket_Address>::value);

Socket_Address_Class
Socket_Address::
classify() const noexcept
  {
    if(this->family() == AF_INET) {
      // Try IPv4.
      return do_classify_ipv4(do_cast_ipv4(this->m_stor.addr4.sin_addr));
    }
    else if(this->family() == AF_INET6) {
      // Try IPv6.
      return do_classify_ipv6(do_cast_ipv6(this->m_stor.addr6.sin6_addr));
    }
    else
      return socket_address_class_reserved;
  }

Socket_Address&
Socket_Address::
parse(const char* host, uint16_t port)
  {
    if(::inet_pton(AF_INET, host, &(this->m_stor.addr4.sin_addr)) == 1) {
      // Try IPv4.
      this->m_stor.addr4.sin_family = AF_INET;
      this->m_stor.addr4.sin_port = htobe16(port);
      this->m_size = sizeof(this->m_stor.addr4);
      return *this;
    }
    else if(::inet_pton(AF_INET6, host, &(this->m_stor.addr6.sin6_addr)) == 1) {
      // Try IPv6.
      this->m_stor.addr6.sin6_family = AF_INET6;
      this->m_stor.addr6.sin6_port = htobe16(port);
      this->m_stor.addr6.sin6_flowinfo = 0;
      this->m_stor.addr6.sin6_scope_id = 0;
      this->m_size = sizeof(this->m_stor.addr6);
      return *this;
    }
    else
      POSEIDON_THROW("Unrecognized host format '$1'", host);
  }

tinyfmt&
Socket_Address::
print(tinyfmt& fmt) const
  {
    if(this->family() == AF_INET) {
      // Try IPv4.
      char sbuf[64];
      const char* host = ::inet_ntop(AF_INET, &(this->m_stor.addr4.sin_addr),
                                     sbuf, sizeof(sbuf));
      if(!host)
        return fmt << "[invalid IPv4 address]";

      return fmt << host << ":" << be16toh(this->m_stor.addr4.sin_port);
    }
    else if(this->family() == AF_INET6) {
      // Try IPv6.
      char sbuf[128];
      const char* host = ::inet_ntop(AF_INET6, &(this->m_stor.addr6.sin6_addr),
                                     sbuf, sizeof(sbuf));
      if(!host)
        return fmt << "[invalid IPv6 address]";

      return fmt << "[" << host << "]:" << be16toh(this->m_stor.addr6.sin6_port);
    }
    else
      return fmt << "[unknown address family `" << this->family() << "`>";
  }

}  // namespace poseidon
