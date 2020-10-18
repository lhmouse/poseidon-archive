// This file is part of Poseidon.
// Copyleft 2020, LH_Mouse. All wrongs reserved.

#include "../precompiled.hpp"
#include "abstract_timer.hpp"
#include "../static/timer_driver.hpp"
#include "../util.hpp"

namespace poseidon {

Abstract_Timer::
~Abstract_Timer()
  {
  }

Abstract_Timer&
Abstract_Timer::
reset(int64_t first, int64_t period)
  noexcept
  {
    Timer_Driver::invalidate_internal(this, first, period);
    return *this;
  }

}  // namespace poseidon
