// This file is part of Poseidon.
// Copyleft 2020, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_CORE_ABSTRACT_FIBER_
#define POSEIDON_CORE_ABSTRACT_FIBER_

#include "../fwd.hpp"

namespace poseidon {

class Abstract_Fiber
  {
  protected:
    friend class Fiber_Scheduler;
    atomic_relaxed<Async_State> m_async_state = { async_state_pending };

  public:
    // Constructs an empty fiber.
    Abstract_Fiber() noexcept;

  protected:
    // This callback is invoked by the fiber scheduler and is intended to be
    // overriden by derived classes.
    virtual
    void
    do_abstract_fiber_on_execution()
      = 0;

  public:
    ASTERIA_NONCOPYABLE_VIRTUAL_DESTRUCTOR(Abstract_Fiber);

    // Gets the schedule state.
    Async_State
    async_state() const noexcept
      { return this->m_async_state.load();  }
  };

}  // namespace poseidon

#endif
