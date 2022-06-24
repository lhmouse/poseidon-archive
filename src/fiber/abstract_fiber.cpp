// This file is part of Poseidon.
// Copyleft 2022, LH_Mouse. All wrongs reserved.

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
do_abstract_fiber_on_suspended()
  {
    POSEIDON_LOG_DEBUG(("Suspending fiber `$1` (class `$2`)"), this, typeid(*this));
  }

void
Abstract_Fiber::
do_abstract_fiber_on_resumed()
  {
    POSEIDON_LOG_DEBUG(("Resuming fiber `$1` (class `$2`)"), this, typeid(*this));
  }

void
Abstract_Fiber::
yield(const shared_ptr<Abstract_Future>& futr_opt) const
  {
    if(!this->m_scheduler)
      POSEIDON_THROW(("Fiber not yieldable unless assigned to a scheduler"));

    // Check that we are yielding within the current fiber.
    this->m_scheduler->checked_yield(this, futr_opt);
  }

}  // namespace poseidon
