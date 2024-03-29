// This file is part of Poseidon.
// Copyleft 2022 - 2023, LH_Mouse. All wrongs reserved.

#include "../precompiled.ipp"
#include "abstract_fiber.hpp"
#include "../static/async_logger.hpp"
#include "../static/fiber_scheduler.hpp"
#include "../utils.hpp"

namespace poseidon {

Abstract_Fiber::
Abstract_Fiber() noexcept
  {
  }

Abstract_Fiber::
~Abstract_Fiber()
  {
  }

void
Abstract_Fiber::
do_abstract_fiber_on_resumed()
  {
    POSEIDON_LOG_TRACE(("Resuming fiber `$1` (class `$2`)"), this, typeid(*this));
  }

void
Abstract_Fiber::
do_abstract_fiber_on_suspended()
  {
    POSEIDON_LOG_TRACE(("Suspending fiber `$1` (class `$2`)"), this, typeid(*this));
  }

void
Abstract_Fiber::
yield(shared_ptrR<Abstract_Future> futr_opt, int64_t fail_timeout_override) const
  {
    if(!this->m_scheduler)
      POSEIDON_THROW(("Fiber not yieldable unless assigned to a scheduler"));

    // Check that we are yielding within the current fiber.
    this->m_scheduler->checked_yield(this, futr_opt, fail_timeout_override);
  }

}  // namespace poseidon
