// This file is part of Poseidon.
// Copyleft 2020, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_CORE_ABSTRACT_TIMER_
#define POSEIDON_CORE_ABSTRACT_TIMER_

#include "../fwd.hpp"

namespace poseidon {

class Abstract_Timer
  {
    friend class Timer_Driver;

  private:
    uint64_t m_serial;
    atomic_relaxed<uint64_t> m_count;

  public:
    // Constructs a timer whose count is zero.
    Abstract_Timer() noexcept;

  protected:
    // This callback is invoked by the timer thread and is intended to be
    // overriden by derived classes. `now` is the number of milliseconds
    // since system startup.
    virtual
    void
    do_abstract_timer_on_tick(int64_t now)
      = 0;

  public:
    ASTERIA_NONCOPYABLE_VIRTUAL_DESTRUCTOR(Abstract_Timer);

    // Gets the number of times that this timer has been triggered.
    size_t
    count() const noexcept
      { return this->m_count.load();  }
  };

}  // namespace

#endif
