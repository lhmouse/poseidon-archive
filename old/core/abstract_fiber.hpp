// This file is part of Poseidon.
// Copyleft 2020, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_CORE_ABSTRACT_FIBER_
#define POSEIDON_CORE_ABSTRACT_FIBER_

#include "../fwd.hpp"
#include <ucontext.h>

namespace poseidon {

class Abstract_Fiber
  {
    friend Fiber_Scheduler;

  private:
    atomic_relaxed<bool> m_cancelled;
    atomic_relaxed<Async_State> m_state;

    // These are scheduler data.
    uint32_t m_sched_serial;
    bool m_sched_running;
    int64_t m_sched_yield_since;
    int64_t m_sched_yield_timeout;

    shared_ptr<Abstract_Future> m_sched_futp;
    ::ucontext_t m_sched_uctx[1];

  protected:
    explicit
    Abstract_Fiber() noexcept
      { }

    POSEIDON_DELETE_COPY(Abstract_Fiber);

  private:
    // Executes this fiber.
    // This function is called only once. No matter whether it returns or
    // throws an exception, this fiber is deleted from the scheduler.
    virtual
    void
    do_abstract_fiber_execute()
      = 0;

    // These are callbacks for profiling.
    // The default implementations do nothing.
    virtual
    void
    do_abstract_fiber_on_start() noexcept;

    virtual
    void
    do_abstract_fiber_on_suspend() noexcept;

    virtual
    void
    do_abstract_fiber_on_resume() noexcept;

    virtual
    void
    do_abstract_fiber_on_finish() noexcept;

  public:
    virtual
    ~Abstract_Fiber();

    // Marks this fiber to be deleted.
    bool
    cancel() noexcept
      { return this->m_cancelled.xchg(true);  }

    // Gets the fiber state, which is set by the scheduler.
    ROCKET_PURE
    Async_State
    state() const noexcept
      { return this->m_state.load();  }
  };

}  // namespace poseidon

#endif
