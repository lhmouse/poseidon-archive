// This file is part of Poseidon.
// Copyleft 2020, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_CORE_ABSTRACT_FUTURE_
#define POSEIDON_CORE_ABSTRACT_FUTURE_

#include "../fwd.hpp"

namespace poseidon {

class Abstract_Future
  {
  protected:
    friend class Fiber_Scheduler;
    once_flag m_once;
    atomic_acq_rel<Future_State> m_future_state = { future_state_empty };

  public:
    // Constructs an empty future.
    Abstract_Future() noexcept;

  public:
    ASTERIA_NONCOPYABLE_VIRTUAL_DESTRUCTOR(Abstract_Future);

    // Gets the future state.
    // This function has acquire semantics. If `future_state_value` is returned,
    // a value should have been initialized by derived classes.
    Future_State
    future_state() const noexcept
      { return this->m_future_state.load();  }
  };

}  // namespace poseidon

#endif
