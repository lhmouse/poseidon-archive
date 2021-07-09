// This file is part of Poseidon.
// Copyleft 2020, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_CORE_ABSTRACT_FUTURE_HPP_
#define POSEIDON_CORE_ABSTRACT_FUTURE_HPP_

#include "../fwd.hpp"

namespace poseidon {

class Abstract_Future
  : public ::asteria::Rcfwd<Abstract_Future>
  {
    template<typename> friend class Promise;
    template<typename> friend class Future;
    friend Fiber_Scheduler;

  private:
    mutable simple_mutex m_mutex;

    // This is a list of fibers that are awaiting this future.
    // It can only be accessed when the scheduler mutex is locked.
    mutable Abstract_Fiber* m_sched_waiting_head = nullptr;

  protected:
    explicit
    Abstract_Future() noexcept
      = default;

  private:
    // Checks whether a value or exception has been set.
    // This functions is called by the fiber scheduler with the global mutex locked.
    ROCKET_PURE bool
    do_is_empty() const noexcept
      { return this->state() == future_state_empty;  }

  public:
    ASTERIA_NONCOPYABLE_DESTRUCTOR(Abstract_Future);

    // Gets the state, which is any of `future_state_empty`, `future_state_value`
    // or `future_state_except`.
    ROCKET_PURE virtual Future_State
    state() const noexcept
      = 0;
  };

}  // namespace poseidon

#endif
