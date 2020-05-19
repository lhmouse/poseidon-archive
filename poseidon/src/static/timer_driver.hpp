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
