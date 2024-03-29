// This file is part of Poseidon.
// Copyleft 2022 - 2023, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_STATIC_TIMER_DRIVER_
#define POSEIDON_STATIC_TIMER_DRIVER_

#include "../fwd.hpp"

namespace poseidon {

// This class schedules timers.
// Objects of this class are recommended to be static.
class Timer_Driver
  {
  private:
    mutable plain_mutex m_pq_mutex;
    condition_variable m_pq_avail;
    uint64_t m_serial;
    struct Queued_Timer;
    vector<Queued_Timer> m_pq;

  public:
    // Constructs an empty driver.
    explicit
    Timer_Driver();

  public:
    ASTERIA_NONCOPYABLE_DESTRUCTOR(Timer_Driver);

    // Gets the current time from a monotonic clock, in nanoseconds.
    // This function is thread-safe.
    static
    int64_t
    clock() noexcept;

    // Schedules timers.
    // This function should be called by the timer thread repeatedly.
    void
    thread_loop();

    // Inserts a timer. If a timer is inserted multiple times, operations other
    // than the last one are invalidated, which can be used to reset a timer.
    // `delay` specifies the number of nanoseconds that a timer will be triggered
    // after it is inserted successfully. `period` is the number of nanoseconds
    // of intervals for periodic timers. `period` can be zero to denote a one-shot
    // timer.
    // This function is thread-safe.
    void
    insert(shared_ptrR<Abstract_Timer> timer, int64_t delay, int64_t period);
  };

}  // namespace poseidon
#endif
