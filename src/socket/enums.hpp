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

// Socket states
enum Socket_State : uint8_t
  {
    socket_state_unknown      = 0,
    socket_state_connecting   = 1,
    socket_state_accepted     = 2,
    socket_state_established  = 3,
    socket_state_closed       = 4,
  };

// Socket I/O results
enum IO_Result : uint8_t
  {
    io_result_partial      = 0,
    io_result_would_block  = 1,
    io_result_end_of_file  = 2,
  };

}  // namespace poseidon

#endif