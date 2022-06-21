// This file is part of Poseidon.
// Copyleft 2022, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_CORE_ENUMS_
#define POSEIDON_CORE_ENUMS_

#include "../fwd.hpp"

namespace poseidon {

// Asynchronous function states
enum Async_State : uint8_t
  {
    async_state_pending    = 0,
    async_state_suspended  = 1,
    async_state_running    = 2,
    async_state_finished   = 3,
  };

// Future states
enum Future_State : uint8_t
  {
    future_state_empty      = 0,
    future_state_value      = 1,
    future_state_exception  = 2,
  };

}  // namespace poseidon

#endif
