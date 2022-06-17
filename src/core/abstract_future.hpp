// This file is part of Poseidon.
// Copyleft 2020, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_CORE_ABSTRACT_FUTURE_
#define POSEIDON_CORE_ABSTRACT_FUTURE_

#include "../fwd.hpp"

namespace poseidon {

class Abstract_Future
  {
  private:
    template<typename> friend class Future;;
    friend class Fiber_Scheduler;

    mutable plain_mutex m_mutex;
    atomic_acq_rel<Future_State> m_state = { future_state_empty };
    vector<weak_ptr<atomic_relaxed<int64_t>>> m_waiters;

  protected:
    // Constructs an empty future.
    Abstract_Future() noexcept;

  private:
    void
    do_abstract_future_check_value(const char* type, const exception_ptr* exptr) const;

    void
    do_abstract_future_signal_nolock() noexcept;

  public:
    ASTERIA_NONCOPYABLE_VIRTUAL_DESTRUCTOR(Abstract_Future);

    // Gets the future state.
    // This function has acquire semantics. If `future_state_value` is returned,
    // a value should have been initialized by derived classes.
    Future_State
    future_state() const noexcept
      { return this->m_state.load();  }

    bool
    empty() const noexcept
      { return this->m_state.load() == future_state_empty;  }

    bool
    has_value() const noexcept
      { return this->m_state.load() == future_state_value;  }

    bool
    has_exception() const noexcept
      { return this->m_state.load() == future_state_exception;  }
  };

}  // namespace poseidon

#endif
