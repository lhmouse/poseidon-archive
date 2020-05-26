// This file is part of Poseidon.
// Copyleft 2020, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_NETWORK_SOCKET_ADDRESS_HPP_
#define POSEIDON_NETWORK_SOCKET_ADDRESS_HPP_

#include "../fwd.hpp"
#include "enums.hpp"
#include <sys/socket.h>
#include <netinet/in.h>

namespace poseidon {

class Socket_Address
  {
  public:
    // For maximum compatibility, this union is recommended to be used
    // in place of the legacy `::sockaddr`.
    union storage_type
      {
        ::sockaddr_storage stor;
        ::sockaddr addr;
        ::sockaddr_in addr4;
        ::sockaddr_in6 addr6;

        constexpr operator
        const void*()
        const noexcept
          { return &(this->addr);  }

        operator
        void*()
        noexcept
          { return &(this->addr);  }

        constexpr operator
        const sockaddr*()
        const noexcept
          { return &(this->addr);  }

        operator
        sockaddr*()
        noexcept
          { return &(this->addr);  }

        constexpr operator
        const sockaddr_in*()
        const noexcept
          { return &(this->addr4);  }

        operator
        sockaddr_in*()
        noexcept
          { return &(this->addr4);  }

        constexpr operator
        const sockaddr_in6*()
        const noexcept
          { return &(this->addr6);  }

        operator
        sockaddr_in6*()
        noexcept
          { return &(this->addr6);  }
      };

    using size_type = ::socklen_t;

  private:
    storage_type m_stor;
    size_type m_size;

  public:
    // Note this is a trivially copyable and destructible class.
    constexpr
    Socket_Address()
    noexcept
      : m_stor(), m_size(0)
      { }

    Socket_Address(const storage_type& stor, size_type size)
    noexcept
      { this->assign(stor, size);  }

    Socket_Address(const char* host, uint16_t port)
      { this->parse(host, port);  }

  public:
    // Gets the `AF_` or `PF_` field.
    ::sa_family_t
    family()
    const noexcept
      { return this->m_stor.addr.sa_family;  }

    // Checks whether this is an IPv4 address.
    // An IPv4-mapped IPv6 address is not an IPv4 one.
    bool
    is_ipv4()
    const noexcept
      { return this->m_stor.addr.sa_family == AF_INET;  }

    // Checks whether this is an IPv6 address.
    // An IPv4-mapped IPv6 address is an IPv6 one.
    bool
    is_ipv6()
    const noexcept
      { return this->m_stor.addr.sa_family == AF_INET6;  }

    // Gets internal data.
    // The pointer and size can be passed to `bind()` or `connect()`
    const storage_type&
    data()
    const noexcept
      { return this->m_stor;  }

    ::socklen_t
    size()
    const noexcept
      { return this->m_size;  }

    // Resets to the default-constructed one (all zeroes).
    Socket_Address&
    clear()
    noexcept
      {
        this->m_stor.addr.sa_family = 0;
        this->m_size = 0;
        return *this;
      }

    Socket_Address&
    swap(Socket_Address& other)
    noexcept
      {
        ::std::swap(this->m_stor, other.m_stor);
        ::std::swap(this->m_size, other.m_size);
        return *this;
      }

    // Classifies this address.
    ROCKET_PURE_FUNCTION
    Address_Class
    classify()
    const noexcept;

    bool
    is_loopback()
    const noexcept
      { return this->classify() == address_class_loopback;  }

    bool
    is_private()
    const noexcept
      { return this->classify() == address_class_private;  }

    bool
    is_multicast()
    const noexcept
      { return this->classify() == address_class_multicast;  }

    bool
    is_public()
    const noexcept
      { return this->classify() == address_class_public;  }

    // Sets contents from the result of a call to `recvfrom()`.
    // Either an IPv4 or IPv6 address may be specified. The address family is
    // detected automatically.
    // This function throws an exception upon failure, and the contents of `*this`
    // is undefined.
    Socket_Address&
    assign(const storage_type& stor, size_type size)
    noexcept
      {
        this->m_stor = stor;
        this->m_size = size;
        return *this;
      }

    // Loads address from a string.
    // Either an IPv4 or IPv6 address may be specified. The address family is
    // detected automatically.
    // This function throws an exception upon failure, and the contents of `*this`
    // is undefined.
    Socket_Address&
    parse(const char* host, uint16_t port);

    // Converts this address to a human-readable string.
    // This is the inverse function of `assign()` except that the port is appended
    // to the host as a string, separated by a colon.
    tinyfmt&
    print(tinyfmt& fmt)
    const;
  };

inline
void
swap(Socket_Address& lhs, Socket_Address& rhs)
noexcept
  { lhs.swap(rhs);  }

inline
tinyfmt&
operator<<(tinyfmt& fmt, const Socket_Address& addr)
  { return addr.print(fmt);  }

}  // namespace poseidon

#endif
