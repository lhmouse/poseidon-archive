// This file is part of Poseidon.
// Copyleft 2022, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_FIBER_ABSTRACT_FIBER_
#define POSEIDON_FIBER_ABSTRACT_FIBER_

#include "../fwd.hpp"

namespace poseidon {

class Abstract_Fiber
  {
  private:
    friend class Fiber_Scheduler;

    Fiber_Scheduler* m_scheduler = nullptr;
    atomic_relaxed<Async_State> m_state = { async_state_pending };

  protected:
    // Constructs an empty fiber.
    explicit
    Abstract_Fiber() noexcept;

  protected:
     // Gets the scheduler instance inside the callbacks hereafter.
     // If this function is called elsewhere, the behavior is undefined.
     Fiber_Scheduler&
     do_abstract_fiber_scheduler() const noexcept
       {
         ROCKET_ASSERT(this->m_scheduler);
         return *(this->m_scheduler);
       }

    // This callback is invoked by the fiber scheduler and is intended to be
    // overriden by derived classes.
    virtual
    void
    do_abstract_fiber_on_execution() = 0;

    // This callback is invoked before `do_abstract_fiber_on_execution()`, and
    // after it is resumed from a preivous yield operation. `async_state()` can
    // be used to examine the current operation. Exceptions are ignored during
    // startup.
    // The default implementations merely print a message.
    virtual
    void
    do_abstract_fiber_on_resumed();

    // This callback is invoked after `do_abstract_fiber_on_execution()`, and
    // before it is suspended by a yield operation. `async_state()` can  be
    // used to examine the current operation. Exceptions are ignored during
    // termination.
    // The default implementations merely print a message.
    void
    do_abstract_fiber_on_suspended();

  public:
    ASTERIA_NONCOPYABLE_VIRTUAL_DESTRUCTOR(Abstract_Fiber);

    // Gets the schedule state.
    Async_State
    async_state() const noexcept
      { return this->m_state.load();  }

    // Calls `m_scheduler->yield(futr_opt)`.
    // This function can only be called from `do_abstract_fiber_on_execution()`.
    void
    yield(const shared_ptr<Abstract_Future>& futr_opt = nullptr, int64_t fail_timeout_override = 0) const;
  };

}  // namespace poseidon
#endif
