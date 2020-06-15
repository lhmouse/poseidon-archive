// This file is part of Poseidon.
// Copyleft 2020, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_STATIC_FIBER_SCHEDULER_HPP_
#define POSEIDON_STATIC_FIBER_SCHEDULER_HPP_

#include "../fwd.hpp"

namespace poseidon {

class Fiber_Scheduler
  {
    POSEIDON_STATIC_CLASS_DECLARE(Fiber_Scheduler);

  public:
    // Executes fibers and blocks until `exit_sig` becomes non-zero.
    // This function is typically called by the main thread. Multiple worker
    // threads are allowed.
    // This function is thread-safe.
    static
    void
    modal_loop(const volatile ::std::atomic<int>& exit_sig);

    // Reloads settings from main config.
    // If this function fails, an exception is thrown, and there is no effect.
    // This function is thread-safe.
    static
    void
    reload();

    // Gets a pointer to the current fiber on the current thread.
    // Its fiber state is `fiber_state_running`.
    // This function is thread-safe.
    ROCKET_PURE_FUNCTION static
    Abstract_Fiber*
    current_opt()
    noexcept;

    // Suspends the current fiber until `promise` becomes satisfied.
    // `current_opt()` must not return null when this function is called. It is
    // taken by value because it has to have a copy on the current call stack.
    // If `futr_opt` is null, the current time slice is relinquished, similar
    // to `sched_yield()`.
    // This function is thread-safe.
    static
    void
    yield(rcptr<const Abstract_Future> futr_opt);

    // Inserts a fiber.
    // This function is thread-safe.
    static
    rcptr<Abstract_Fiber>
    insert(uptr<Abstract_Fiber>&& ufiber);
  };

}  // namespace poseidon

#endif