// This file is part of Poseidon.
// Copyleft 2022, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_CORE_ABSTRACT_FIBER_
#define POSEIDON_CORE_ABSTRACT_FIBER_

#include "../fwd.hpp"
#include "enums.hpp"

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
    // This callback is invoked by the fiber scheduler and is intended to be
    // overriden by derived classes.
    virtual
    void
    do_abstract_fiber_on_execution()
      = 0;

    // These callbacks are called when a fiber is suspended and resumed. They
    // are not called when the fiber starts execution or terminates.
    // The default implementations merely print a message.
    virtual
    void
    do_abstract_fiber_on_suspended();

    virtual
    void
    do_abstract_fiber_on_resumed();

  public:
    ASTERIA_NONCOPYABLE_VIRTUAL_DESTRUCTOR(Abstract_Fiber);

    // Gets the schedule state.
    Async_State
    async_state() const noexcept
      { return this->m_state.load();  }

    // Calls `m_scheduler->yield(futr_opt)`.
    // This function can only be called from `do_abstract_fiber_on_execution()`.
    void
    yield(const shared_ptr<Abstract_Future>& futr_opt = nullptr) const;
  };

}  // namespace poseidon

#endif
