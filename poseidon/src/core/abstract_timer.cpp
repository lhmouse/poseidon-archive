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
do_on_async_timer(int64_t /*now*/)
const
  {
    POSEIDON_LOG_ERROR("TODO: Please override `do_on_async_timer()` in `$1`",
                       typeid(*this).name());
  }

void
Abstract_Timer::
reset(int64_t first, int64_t period)
noexcept
  {
    // Update data members.
    ::rocket::mutex::unique_lock lock(this->m_mutex);
    this->m_first = first;
    this->m_period = period;
    lock.unlock();

    // Notify the driver about the update.
    Timer_Driver::invalidate_internal(this);
  }

}  // namespace poseidon
