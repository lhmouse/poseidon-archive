// This file is part of Poseidon.
// Copyleft 2020, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_NETWORK_ENUMS_HPP_
#define POSEIDON_NETWORK_ENUMS_HPP_

#include "../fwd.hpp"

namespace poseidon {

// This is the return type of I/O functions.
// `io_result_not_eof` may be returned to indicate success of a
// non-stream operation, such as `accept()` or `recvfrom()`.
// I/O functions shall throw exceptions for errors that are not
// listed here.
// Note that positive values denote number of bytes transferred.
enum IO_Result : ptrdiff_t
  {
    io_result_again    = -1,  // EAGAIN or EWOULDBLOCK
    io_result_eof      =  0,
    io_result_not_eof  =  PTRDIFF_MAX,  // EINTR
  };

// This describes the lifetime of a connection.
// It is mainly designed for stream-oriented protocols such as
// TCP and SCTP.
enum Connection_State : uint8_t
  {
    connection_state_initial      = 0,  // W allowed
    connection_state_connecting   = 1,  // W allowed
    connection_state_established  = 2,  // R/W allowed
    connection_state_closing      = 3,  // R/W forbidden
    connection_state_closed       = 4,  // R/W forbidden
  };

// This classifies IP addresses.
// These values are shared by IPv4 and IPv6.
enum Address_Class : uint8_t
  {
    address_class_reserved   = 0,
    address_class_loopback   = 1,
    address_class_private    = 2,
    address_class_multicast  = 3,
    address_class_public     = 4,
  };

}  // namespace poseidon

#endif
