// This file is part of Poseidon.
// Copyleft 2020, LH_Mouse. All wrongs reserved.

#include "../precompiled.hpp"
#include "abstract_timer.hpp"
#include "../static/timer_driver.hpp"
#include "../utilities.hpp"

namespace poseidon {

Abstract_Timer::
~Abstract_Timer()
  {
  }

void
Abstract_Timer::
reset(int64_t first, int64_t period)
noexcept
  {
    Timer_Driver::invalidate_internal(this, first, period);
  }

}  // namespace poseidon
