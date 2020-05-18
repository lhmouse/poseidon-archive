// This file is part of Poseidon.
// Copyleft 2020, LH_Mouse. All wrongs reserved.

#include "../precompiled.hpp"
#include "socket_address.hpp"
#include "../utilities.hpp"
#include <arpa/inet.h>

namespace poseidon {
namespace {

inline
::std::bitset<32>
do_cast_ipv4(const ::in_addr& in4)
noexcept
  {
    ::std::bitset<32> bits;
    bits = be32toh(in4.s_addr);
    return bits;
  }

constexpr
::std::bitset<32>
do_make_ipv4(const array<uint8_t, 4>& segs)
noexcept
  {
    ::std::bitset<32> bits;
    for(size_t k = 0;  k != 4;  ++k)
      bits |= ::std::bitset<32>(segs[k]) << (24 - k * 8);
    return bits;
  }

inline
::std::bitset<128>
do_cast_ipv6(const ::in6_addr& in6)
noexcept
  {
    ::std::bitset<128> bits;
    for(size_t k = 0;  k != 16;  ++k)
      bits |= ::std::bitset<128>(in6.s6_addr[k]) << (120 - k * 8);
    return bits;
  }

constexpr
::std::bitset<128>
do_make_ipv6(const array<uint16_t, 8>& segs)
noexcept
  {
    ::std::bitset<128> bits;
    for(size_t k = 0;  k != 8;  ++k)
      bits |= ::std::bitset<128>(segs[k]) << (112 - k * 16);
    return bits;
  }

template<size_t N>
constexpr
bool
do_match(const ::std::bitset<N>& addr, const ::std::bitset<N>& comp, size_t bits)
noexcept
  {
    ::std::bitset<N> mask;
    if(bits < mask.size())
      mask.set() >>= bits;  // shift in zeroes
    mask.flip();

    ROCKET_ASSERT((comp & mask) == comp);
    return (addr & mask) == comp;
  }

ROCKET_PURE_FUNCTION
bool
do_is_ipv4_private(const ::std::bitset<32>& addr)
noexcept
  {
    // 0.0.0.0/8: Local Identification
    if(do_match(addr, do_make_ipv4({0,0,0,0}), 8))
      return true;

    // 10.0.0.0/8: Class A Private-Use
    if(do_match(addr, do_make_ipv4({10,0,0,0}), 8))
      return true;

    // 127.0.0.0/8: Loopback
    if(do_match(addr, do_make_ipv4({127,0,0,0}), 8))
      return true;

    // 172.16.0.0/12: Class B Private-Use
    if(do_match(addr, do_make_ipv4({172,16,0,0}), 12))
      return true;

    // 169.254.0.0/16: Link Local
    if(do_match(addr, do_make_ipv4({169,254,0,0}), 16))
      return true;

    // 192.168.0.0/16: Class C Private-Use
    if(do_match(addr, do_make_ipv4({192,168,0,0}), 16))
      return true;

    // 224.0.0.0/4: Class D
    if(do_match(addr, do_make_ipv4({224,0,0,0}), 4))
      return true;

    // 240.0.0.0/4: Class E
    if(do_match(addr, do_make_ipv4({240,0,0,0}), 4))
      return true;

    return false;
  }

ROCKET_PURE_FUNCTION
bool
do_is_ipv6_private(const ::std::bitset<128>& addr)
noexcept
  {
    // ::/128: Unspecified
    if(do_match(addr, do_make_ipv6({0,0,0,0,0,0,0,0}), 128))
      return true;

    // ::1/128: Loopback
    if(do_match(addr, do_make_ipv6({0,0,0,0,0,0,0,1}), 128))
      return true;

    // ::ffff:0:0/96: IPv4-mapped
    if(do_match(addr, do_make_ipv6({0,0,0,0,0,0xffff,0,0}), 96))
      return do_is_ipv4_private(addr.to_ulong());

    // 64:ff9b::/96: IPv4 to IPv6
    if(do_match(addr, do_make_ipv6({0x64,0xff9b,0,0,0,0,0,0}), 96))
      return do_is_ipv4_private(addr.to_ulong());

    // 64:ff9b:1::/48: Local-Use IPv4/IPv6
    if(do_match(addr, do_make_ipv6({0x64,0xff9b,1,0,0,0,0,0}), 48))
      return true;

    // 100::/64: Discard-Only
    if(do_match(addr, do_make_ipv6({0x100,0,0,0,0,0,0,0}), 64))
      return true;

    // 2001:db8::/32: Documentation
    if(do_match(addr, do_make_ipv6({0x2001,0xdb8,0,0,0,0,0,0}), 32))
      return true;

    // 2002::/16: 6to4
    if(do_match(addr, do_make_ipv6({0x2002,0,0,0,0,0,0,0}), 16))
      return do_is_ipv4_private((addr >> 80).to_ulong());

    return false;
  }

}  // namespace

static_assert(::std::is_trivially_copy_constructible<Socket_Address>::value &&
              ::std::is_trivially_move_constructible<Socket_Address>::value &&
              ::std::is_trivially_destructible<Socket_Address>::value);

bool
Socket_Address::
is_private()
const noexcept
  {
    if(this->family() == AF_INET) {
      // Try IPv4.
      return do_is_ipv4_private(do_cast_ipv4(this->m_stor.addr4.sin_addr));
    }
    else if(this->family() == AF_INET6) {
      // Try IPv6.
      return do_is_ipv6_private(do_cast_ipv6(this->m_stor.addr6.sin6_addr));
    }
    else
      return false;
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
      POSEIDON_THROW("unrecognized host format: $1", host);
  }

tinyfmt&
Socket_Address::
print(tinyfmt& fmt)
const
  {
    if(this->family() == AF_INET) {
      // Try IPv4.
      char sbuf[64];
      const char* host = ::inet_ntop(AF_INET, &(this->m_stor.addr4.sin_addr),
                                     sbuf, sizeof(sbuf));
      if(!host)
        return fmt << "<invalid IPv4 address>";

      return fmt << host << ":" << be16toh(this->m_stor.addr4.sin_port);
    }
    else if(this->family() == AF_INET6) {
      // Try IPv6.
      char sbuf[128];
      const char* host = ::inet_ntop(AF_INET6, &(this->m_stor.addr6.sin6_addr),
                                     sbuf, sizeof(sbuf));
      if(!host)
        return fmt << "<invalid IPv6 address>";

      return fmt << "[" << host << "]:" << be16toh(this->m_stor.addr6.sin6_port);
    }
    else
      return fmt << "<unknown address family" << this->family() << ">";
  }

}  // namespace poseidon
