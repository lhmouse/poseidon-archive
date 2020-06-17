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
    ::std::atomic<bool> m_resident;  // don't delete if orphaned
    ::std::atomic<uint64_t> m_count;

    ::std::atomic<int64_t> m_first;  // absolute time in milliseconds
    ::std::atomic<int64_t> m_period;  // period in milliseconds

  public:
    Abstract_Timer(int64_t first, int64_t period)
    noexcept
      : m_resident(false), m_count(0),
        m_first(first), m_period(period)
      { }

    ASTERIA_NONCOPYABLE_DESTRUCTOR(Abstract_Timer);

  protected:
    // `now` is the time of `CLOCK_MONOTONIC`.
    // Please mind thread safety, as this function is called by the timer thread.
    // The default implementation prints a line of error.
    virtual
    void
    do_on_async_timer(int64_t now)
      = 0;

  public:
    // Should this timer be deleted if timer driver holds its last reference?
    ROCKET_PURE_FUNCTION
    bool
    resident()
    const noexcept
      { return this->m_resident.load(::std::memory_order_relaxed);  }

    void
    set_resident(bool value = true)
    noexcept
      { this->m_resident.store(value, ::std::memory_order_relaxed);  }

    // Gets the counter.
    ROCKET_PURE_FUNCTION
    uint64_t
    count()
    const noexcept
      { return this->m_count.load(::std::memory_order_relaxed);  }

    // Resets the first triggered time and the period.
    void
    reset(int64_t first, int64_t period)
    noexcept;
  };

}  // namespace poseidon

#endif
