// This file is part of Poseidon.
// Copyleft 2022, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_SOCKET_ENUMS_
#define POSEIDON_SOCKET_ENUMS_

#include "../fwd.hpp"

namespace poseidon {

enum IP_Address_Class : uint8_t
  {
    ip_address_class_unspecified  = 0,  // all zeroes
    ip_address_class_reserved     = 1,
    ip_address_class_public       = 2,
    ip_address_class_loopback     = 3,
    ip_address_class_private      = 4,
    ip_address_class_link_local   = 5,
    ip_address_class_multicast    = 6,
    ip_address_class_broadcast    = 7,  // IPv4 only
  };

enum Socket_State : uint8_t
  {
    socket_state_unknown      = 0,
    socket_state_connecting   = 1,
    socket_state_established  = 2,
    socket_state_closing      = 3,
    socket_state_closed       = 4,
  };

}  // namespace poseidon
#endif
