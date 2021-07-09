// This file is part of Poseidon.
// Copyleft 2020, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_CORE_ABSTRACT_TIMER_HPP_
#define POSEIDON_CORE_ABSTRACT_TIMER_HPP_

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

  protected:
    // `now` is the time of `CLOCK_MONOTONIC`.
    // Please mind thread safety, as this function is called by the timer thread.
    // The default implementation prints a line of error.
    virtual void
    do_on_async_timer(int64_t now)
      = 0;

  public:
    ASTERIA_NONCOPYABLE_DESTRUCTOR(Abstract_Timer);

    // Marks this timer to be deleted.
    bool
    shut_down() noexcept
      { return this->m_zombie.exchange(true);  }

    // Marks this this timer to be deleted if timer driver holds its last reference.
    bool
    set_resident(bool value = true) noexcept
      { return this->m_resident.exchange(value);  }

    // Gets the counter.
    ROCKET_PURE uint64_t
    count() const noexcept
      { return this->m_count.load();  }

    // Resets the first triggered time and the period.
    Abstract_Timer&
    reset(int64_t first, int64_t period) noexcept;
  };

}  // namespace poseidon

#endif
