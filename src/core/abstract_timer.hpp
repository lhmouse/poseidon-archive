// This file is part of Poseidon.
// Copyleft 2020, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_CORE_ABSTRACT_TIMER_
#define POSEIDON_CORE_ABSTRACT_TIMER_

#include "../fwd.hpp"

namespace poseidon {

class Abstract_Timer
  : public ::asteria::Rcfwd<Abstract_Timer>
  {
    friend Timer_Driver;

  private:
    atomic_relaxed<bool> m_zombie;
    atomic_relaxed<bool> m_resident;  // don't delete if orphaned
    atomic_relaxed<uint64_t> m_count;

    // These are updated by the timer driver.
    int64_t m_first;
    int64_t m_period;

  protected:
    explicit
    Abstract_Timer(int64_t first, int64_t period) noexcept
      : m_first(first), m_period(period)
      { }

  private:
    // `now` is the time of `CLOCK_MONOTONIC` when this timer is triggered.
    // Please mind thread safety, as this function is called by the timer thread.
    virtual
    void
    do_abstract_timer_interval(int64_t now)
      = 0;

  public:
    ASTERIA_NONCOPYABLE_DESTRUCTOR(Abstract_Timer);

    // Marks this timer to be deleted.
    bool
    shut_down() noexcept
      { return this->m_zombie.xchg(true);  }

    // Prevents this this timer from being deleted if timer driver holds its
    // last reference.
    bool
    set_resident(bool value = true) noexcept
      { return this->m_resident.xchg(value);  }

    // Gets the counter.
    ROCKET_PURE
    uint64_t
    count() const noexcept
      { return this->m_count.load();  }

    // Resets the first triggered time and the period.
    Abstract_Timer&
    reset(int64_t first, int64_t period) noexcept;
  };

}  // namespace poseidon

#endif
