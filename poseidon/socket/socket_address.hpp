// This file is part of Poseidon.
// Copyleft 2022, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_SOCKET_SOCKET_ADDRESS_
#define POSEIDON_SOCKET_SOCKET_ADDRESS_

#include "../fwd.hpp"
#include "enums.hpp"
#include <netinet/in.h>

namespace poseidon {

class Socket_Address
  {
  private:
    ::in6_addr m_addr;
    uint16_t m_port;

  public:
    // Initializes an unspecified address.
    explicit constexpr
    Socket_Address() noexcept
      : m_addr(), m_port(0)
      { }

    // Initializes an address from a foreign source.
    constexpr
    Socket_Address(const ::in6_addr& addr, uint16_t port)
      : m_addr(addr), m_port(port)
      { }

    // Parses an address from a string, like `parse()`.
    // An exception is thrown if the address string is not valid.
    explicit
    Socket_Address(const cow_string& str);

  public:
    // Accesses raw data.
    constexpr
    const ::in6_addr&
    data() const noexcept
      { return this->m_addr;  }

    constexpr
    uint16_t
    port() const noexcept
      { return this->m_port;  }

    ::in6_addr&
    mut_data() noexcept
      { return this->m_addr;  }

    void
    set_data(const ::in6_addr& addr) noexcept
      { this->m_addr = addr;  }

    void
    set_port(uint16_t port) noexcept
      { this->m_port = port;  }

    Socket_Address&
    clear() noexcept
      {
        this->m_addr = ::in6_addr();
        this->m_port = 0;
        return *this;
      }

    Socket_Address&
    swap(Socket_Address& other) noexcept
      {
        ::std::swap(this->m_addr, other.m_addr);
        ::std::swap(this->m_port, other.m_port);
        return *this;
      }

    // Returns the address class, which is shared by both IPv4 and IPv6.
    ROCKET_PURE
    Socket_Address_Class
    classify() const noexcept;

    // Parses an address from a string, which may be an IPv4 address, or
    // an IPv6 address in brackets, followed by an optional port number.
    // Examples are `127.0.0.1:80`, `192.168.1.0` and `[::1]:1300`.
    // If `false` is returned or an exception is thrown, the contents of
    // this object are unspecified.
    bool
    parse(const cow_string& str);

    // Converts this address to its string form.
    tinyfmt&
    print(tinyfmt& fmt) const;

    cow_string
    print_to_string() const;
  };

inline
void
swap(Socket_Address& lhs, Socket_Address& rhs) noexcept
  {
    lhs.swap(rhs);
  }

inline
tinyfmt&
operator<<(tinyfmt& fmt, const Socket_Address& saddr)
  {
    return saddr.print(fmt);
  }

}  // namespace poseidon
#endif
