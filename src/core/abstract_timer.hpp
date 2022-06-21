// This file is part of Poseidon.
// Copyleft 2022, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_CORE_ABSTRACT_TIMER_
#define POSEIDON_CORE_ABSTRACT_TIMER_

#include "../fwd.hpp"
#include "enums.hpp"

namespace poseidon {

class Abstract_Timer
  {
  private:
    friend class Timer_Driver;

    atomic_relaxed<Async_State> m_state = { async_state_pending };
    atomic_relaxed<uint64_t> m_count = { 0 };
    uint64_t m_serial;  // used by timer driver

  protected:
    // Constructs a timer whose count is zero.
    explicit
    Abstract_Timer() noexcept;

  protected:
    // This callback is invoked by the timer thread and is intended to be
    // overriden by derived classes. `now` is the number of nanoseconds since
    // system startup.
    virtual
    void
    do_abstract_timer_on_tick(int64_t now)
      = 0;

  public:
    ASTERIA_NONCOPYABLE_VIRTUAL_DESTRUCTOR(Abstract_Timer);

    // Gets the schedule state.
    Async_State
    async_state() const noexcept
      { return this->m_state.load();  }

    // Gets the number of times that this timer has been triggered.
    uint64_t
    count() const noexcept
      { return this->m_count.load();  }
  };

}  // namespace poseidon

#endif
