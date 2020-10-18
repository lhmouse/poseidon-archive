// This file is part of Poseidon.
// Copyleft 2020, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_STATIC_TIMER_DRIVER_HPP_
#define POSEIDON_STATIC_TIMER_DRIVER_HPP_

#include "../fwd.hpp"

namespace poseidon {

class Timer_Driver
  {
    POSEIDON_STATIC_CLASS_DECLARE(Timer_Driver);

  public:
    // Creates the timer thread if one hasn't been created.
    static
    void
    start();

    // Inserts a timer.
    // The driver holds a reference-counted pointer to the timer. If it becomes a unique
    // reference, the timer is deleted.
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
    invalidate_internal(const Abstract_Timer* timer, int64_t first, int64_t period)
      noexcept;
  };

}  // namespace poseidon

#endif
