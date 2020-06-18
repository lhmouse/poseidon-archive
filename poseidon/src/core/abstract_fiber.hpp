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
    ::std::atomic<bool> m_resident;  // don't delete if orphaned
    ::std::atomic<Async_State> m_state;

    // These are scheduler data.
    uint32_t m_sched_version;
    int64_t m_sched_yield_time;
    const Abstract_Future* m_sched_futp;
    Abstract_Fiber* m_sched_next;
    ::ucontext_t m_sched_uctx[1];

  public:
    Abstract_Fiber()
    noexcept
      : m_resident(false), m_state(async_state_initial)
      { }

    ASTERIA_NONCOPYABLE_DESTRUCTOR(Abstract_Fiber);

  private:
    void
    do_set_state(Async_State state)
    noexcept
      { this->m_state.store(state, ::std::memory_order_release);  }

  protected:
    // Executes this fiber.
    // This function is called only once. No matter whether it returns or
    // throws an exception, this fiber is deleted from the scheduler.
    virtual
    void
    do_execute()
      = 0;

    // These are callbacks for profiling.
    // The default implementations do nothing.
    virtual
    void
    do_on_start()
    noexcept;

    virtual
    void
    do_on_suspend()
    noexcept;

    virtual
    void
    do_on_resume()
    noexcept;

    virtual
    void
    do_on_finish()
    noexcept;

  public:
    // Should this timer be deleted if timer driver holds its last reference?
    ROCKET_PURE_FUNCTION
    bool
    resident()
    const noexcept
      { return this->m_resident.load(::std::memory_order_relaxed);  }

    void
    set_resident(bool value = true)
    noexcept
      { this->m_resident.store(value, ::std::memory_order_relaxed);  }

    // Gets the fiber state, which is set by the scheduler.
    ROCKET_PURE_FUNCTION
    Async_State
    state()
    const noexcept
      { return this->m_state.load(::std::memory_order_acquire);  }
  };

}  // namespace poseidon

#endif
