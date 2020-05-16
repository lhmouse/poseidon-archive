// This file is part of Poseidon.
// Copyleft 2020, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_STATIC_TIMER_DRIVER_HPP_
#define POSEIDON_STATIC_TIMER_DRIVER_HPP_

#include "../fwd.hpp"

namespace poseidon {

class Timer_Driver
  {
    POSEIDON_STATIC_CLASS_DECLARE(Timer_Driver);

  private:
    static
    void
    do_thread_loop(void* param);

  public:
    // Creates the timer thread if one hasn't been created.
    static
    void
    start();

    // Gets the absolute time of the monotonic clock used by all timers.
    // All values are measured in milliseconds. This function adds `shift` to the value
    // with saturation semantics. The result is always non-negative.
    // This function is thread-safe.
    ROCKET_PURE_FUNCTION
    static
    int64_t
    get_tick_count(int64_t shift = 0)
    noexcept;

    // Inserts a timer.
    // The driver holds a reference-counted pointer to the timer. If it becomes a unique
    // reference, the timer is deleted.
    // To prevent the timer from being deleted, call `.release()` of the returned pointer.
    // If this function fails, an exception is thrown, and there is no effect.
    // This function is thread-safe.
    static
    rcptr<Abstract_Timer>
    insert(uptr<Abstract_Timer>&& utimer);

    // Notifies the timer thread that a timer has been updated.
    // This is an internal function and might be inefficient. You will not want to call it.
    // This function is thread-safe.
    static
    bool
    invalidate_internal(Abstract_Timer* timer)
    noexcept;
  };

}  // namespace poseidon

#endif
