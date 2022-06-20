// This file is part of Poseidon.
// Copyleft 2020, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_SOCKET_SOCKET_ADDRESS_
#define POSEIDON_SOCKET_SOCKET_ADDRESS_

#include "../fwd.hpp"
#include "enums.hpp"
#include <sys/socket.h>
#include <netinet/in.h>

namespace poseidon {

class Socket_Address
  {
  private:
    union {
      ::sa_family_t m_family;
      ::sockaddr m_addr;
      ::sockaddr_in m_addr4;
      ::sockaddr_in6 m_addr6;
      ::sockaddr_storage m_stor;
      char m_data[1];
    };
    ::socklen_t m_size;

  public:
    // Initializes an invalid address.
    explicit constexpr
    Socket_Address() noexcept
      : m_family(AF_UNSPEC), m_size(0)
      { }

    // Parses an address from a string.
    explicit
    Socket_Address(const char* host, uint16_t port);

    explicit
    Socket_Address(const cow_string& host, uint16_t port);

  public:
    // Get the address family.
    constexpr
    ::sa_family_t
    family() const noexcept
      { return this->m_family;  }

    constexpr
    bool
    is_ipv4() const noexcept
      { return this->m_family == PF_INET;  }

    constexpr
    bool
    is_ipv6() const noexcept
      { return this->m_family == PF_INET6;  }

    // Get raw data and size. These functions are provided for convenience.
    constexpr
    const void*
    data() const noexcept
      { return this->m_data;  }

    void*
    mut_data() noexcept
      { return this->m_data;  }

    constexpr
    const ::sockaddr*
    addr() const noexcept
      { return &(this->m_addr);  }

    ::sockaddr*
    mut_addr() noexcept
      { return &(this->m_addr);  }

    constexpr
    const ::sockaddr_in*
    addr4() const noexcept
      { return &(this->m_addr4);  }

    ::sockaddr_in*
    mut_addr4() noexcept
      { return &(this->m_addr4);  }

    constexpr
    const ::sockaddr_in6*
    addr6() const noexcept
      { return &(this->m_addr6);  }

    ::sockaddr_in6*
    mut_addr6() noexcept
      { return &(this->m_addr6);  }

    constexpr
    size_t
    size() const noexcept
      { return (::std::make_unsigned<::socklen_t>::type) this->m_size;  }

    constexpr
    ::socklen_t
    ssize() const noexcept
      { return this->m_size;  }

    void
    set_size(::socklen_t nbytes) noexcept
      {
        ROCKET_ASSERT(nbytes <= (::socklen_t) sizeof(this->m_stor));
        this->m_size = nbytes;
      }

    constexpr
    size_t
    capacity() const noexcept
      { return sizeof(this->m_stor);  }

    // These are general modifiers.
    Socket_Address&
    clear() noexcept
      {
        this->m_family = AF_UNSPEC;
        this->m_size = 0;
        return *this;
      }

    Socket_Address&
    swap(Socket_Address& other) noexcept
      {
        ::std::swap(this->m_family, other.m_family);
        ::std::swap(this->m_stor, other.m_stor);
        ::std::swap(this->m_size, other.m_size);
        return *this;
      }

    // Returns the address class, which is shared by both IPv4 and IPv6.
    ROCKET_PURE
    Socket_Address_Class
    classify() const noexcept;

    // Converts this address to its string form.
    tinyfmt&
    print(tinyfmt& fmt) const;

    cow_string
    format() const;

    // Parses an address from a string, which may be either IPv4 or IPv6.
    // In case of failure, an exception is thrown, and the contents of this
    // object are unchanged.
    Socket_Address&
    parse(const char* host, uint16_t port);

    Socket_Address&
    parse(const cow_string& host, uint16_t port);
  };

inline
void
swap(Socket_Address& lhs, Socket_Address& rhs) noexcept
  {
    lhs.swap(rhs);
  }

inline
tinyfmt&
operator<<(tinyfmt& fmt, const Socket_Address& addr)
  {
    return addr.print(fmt);
  }

}  // namespace poseidon

#endif
