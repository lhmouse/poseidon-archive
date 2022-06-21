// This file is part of Poseidon.
// Copyleft 2022, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_SOCKET_ENUMS_
#define POSEIDON_SOCKET_ENUMS_

#include "../fwd.hpp"

namespace poseidon {

// Socket address classes (IPv4 and IPv6 only)
enum Socket_Address_Class : uint8_t
  {
    socket_address_class_unknown    = 0,
    socket_address_class_reserved   = 1,
    socket_address_class_loopback   = 2,
    socket_address_class_private    = 3,
    socket_address_class_multicast  = 4,
    socket_address_class_public     = 5,
  };

}  // namespace poseidon

#endif
