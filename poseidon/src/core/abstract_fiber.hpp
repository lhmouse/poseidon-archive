// This file is part of Poseidon.
// Copyleft 2020, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_CORE_ABSTRACT_FIBER_HPP_
#define POSEIDON_CORE_ABSTRACT_FIBER_HPP_

#include "../fwd.hpp"
#include <ucontext.h>

namespace poseidon {

class Abstract_Fiber
  : public ::asteria::Rcfwd<Abstract_Fiber>
  {
    friend Fiber_Scheduler;

  private:
    ::std::atomic<Async_State> m_state;

    // These are scheduler data.
    Abstract_Fiber* m_sched_next;
    Abstract_Fiber* m_sched_prev;

    int64_t m_sched_time;
    int64_t m_sched_warn;
    const Abstract_Future* m_sched_futr;
    ::ucontext_t m_sched_uctx[1];

  public:
    Abstract_Fiber()
    noexcept
      : m_state(async_state_initial)
      { }

    ASTERIA_NONCOPYABLE_DESTRUCTOR(Abstract_Fiber);

  protected:
    // Executes this fiber.
    // This function is called only once. No matter whether it returns or
    // throws an exception, this fiber is deleted from the scheduler.
    virtual
    void
    do_execute()
      = 0;

  public:
    // Gets the fiber state, which is set by the scheduler.
    ROCKET_PURE_FUNCTION
    Async_State
    state()
    const noexcept
      { return this->m_state.load(::std::memory_order_acquire);  }
  };

}  // namespace poseidon

#endif
