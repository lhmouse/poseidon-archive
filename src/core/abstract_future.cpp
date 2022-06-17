// This file is part of Poseidon.
// Copyleft 2022, LH_Mouse. All wrongs reserved.

#include "../precompiled.ipp"
#include "abstract_future.hpp"
#include "../static/fiber_scheduler.hpp"
#include "../utils.hpp"

namespace poseidon {

Abstract_Future::
Abstract_Future() noexcept
  {
  }

Abstract_Future::
~Abstract_Future()
  {
  }

void
Abstract_Future::
do_abstract_future_check_value(const char* type, const exception_ptr* exptr) const
  {
    switch(this->m_state.load()) {
      case future_state_empty:
        POSEIDON_THROW((
            "No value set",
            "[value type was `$1`]"),
            type);

      case future_state_value:
        break;

      case future_state_exception:
        // `exptr` shall point to an initialized exception pointer here.
        if(*exptr)
          rethrow_exception(*exptr);

        POSEIDON_THROW((
            "Promise brkoen without an exception",
            "[value type was `$1`]"),
            type);
    }
  }

void
Abstract_Future::
do_abstract_future_signal_nolock() noexcept
  {
    for(const auto& wp : this->m_waiters)
      if(auto timep = wp.lock())
        timep->store(Fiber_Scheduler::clock());
  }

}  // namespace poseidon
