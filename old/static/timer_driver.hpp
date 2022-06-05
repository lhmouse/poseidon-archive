// This file is part of Poseidon.
// Copyleft 2020, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_STATIC_TIMER_DRIVER_
#define POSEIDON_STATIC_TIMER_DRIVER_

#include "../fwd.hpp"

namespace poseidon {

class Timer_Driver
  {
    POSEIDON_STATIC_CLASS_DECLARE(Timer_Driver);

  public:
    // Inserts a timer. The driver holds a weak pointer to the timer.
    // If this function fails, an exception is thrown, and there is no effect.
    // This function is thread-safe.
    static
    shared_ptr<Abstract_Timer>
    insert(const shared_ptr<Abstract_Timer>& timer);
  };

}  // namespace poseidon

#endif
