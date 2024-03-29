// This file is part of Poseidon.
// Copyleft 2022 - 2023, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_FIBER_ENUMS_
#define POSEIDON_FIBER_ENUMS_

#include "../fwd.hpp"

namespace poseidon {

enum Future_State : uint8_t
  {
    future_state_empty      = 0,
    future_state_value      = 1,
    future_state_exception  = 2,
  };

}  // namespace poseidon
#endif
