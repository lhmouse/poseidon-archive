// This file is part of Poseidon.
// Copyleft 2020, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_CORE_ABSTRACT_FUTURE_
#define POSEIDON_CORE_ABSTRACT_FUTURE_

#include "../fwd.hpp"

namespace poseidon {

class Abstract_Future
  : public ::asteria::Rcfwd<Abstract_Future>
  {
    template<typename> friend class Promise;
    template<typename> friend class Future;
    friend Fiber_Scheduler;

  private:
    atomic_acq_rel<Future_State> m_state = { future_state_empty };
    mutable once_flag m_once;

    // These are scheduler data.
    ::rocket::cow_vector<rcptr<Abstract_Fiber>> m_sched_sleep_q;

  protected:
    explicit
    Abstract_Future() noexcept
      = default;

  public:
    ASTERIA_NONCOPYABLE_DESTRUCTOR(Abstract_Future);

    // Gets the state, which is any of `future_state_empty`, `future_state_value`
    // or `future_state_except`.
    //
    // * `future_state_empty` indicates no value has been set yet. Any retrieval
    //   operation shall block.
    // * `future_state_value` indicates a value has been set and can be read. Any
    //   retrieval operation shall unblock and return the value.
    // * `future_state_except` indicates either an exception has been set or the
    //   associated promise has gone out of scope without setting a value. Any
    //   retrieval operation shall unblock and throw an exception.
    ROCKET_PURE
    Future_State
    state() const noexcept
      { return this->m_state.load();  }
  };

}  // namespace poseidon

#endif
