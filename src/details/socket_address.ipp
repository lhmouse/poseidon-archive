// This file is part of Poseidon.
// Copyleft 2020, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_SOCKET_SOCKET_ADDRESS_HPP_
#  error Please include <poseidon/socket/socket_address.hpp> instead.
#endif

namespace poseidon {
namespace details_socket_address {

union storage
  {
    ::sockaddr_storage stor;
    ::sockaddr addr;
    ::sockaddr_in addr4;
    ::sockaddr_in6 addr6;

    // This union is implicit convertible to `void*`.
    constexpr operator
    const void*() const noexcept
      { return &(this->addr);  }

    operator
    void*() noexcept
      { return &(this->addr);  }

    // This union is implicit convertible to `sockaddr*`.
    constexpr operator
    const ::sockaddr*() const noexcept
      { return &(this->addr);  }

    operator
    ::sockaddr*() noexcept
      { return &(this->addr);  }

    // This union is implicit convertible to an IPv4 address.
    constexpr operator
    const ::sockaddr_in*() const noexcept
      { return &(this->addr4);  }

    operator
    ::sockaddr_in*() noexcept
      { return &(this->addr4);  }

    // This union is implicit convertible to an IPv6 address.
    constexpr operator
    const ::sockaddr_in6*() const noexcept
      { return &(this->addr6);  }

    operator
    ::sockaddr_in6*() noexcept
      { return &(this->addr6);  }
  };

}  // namespace details_socket_address
}  // namespace poseidon
