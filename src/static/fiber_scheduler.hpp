// This file is part of Poseidon.
// Copyleft 2022, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_STATIC_FIBER_SCHEDULER_
#define POSEIDON_STATIC_FIBER_SCHEDULER_

#include "../fwd.hpp"
#include "../core/config_file.hpp"
#include <ucontext.h>  // ucontext_t

namespace poseidon {

// This class schedules fibers.
// Objects of this class are recommended to be static.
class Fiber_Scheduler
  {
  private:
    mutable plain_mutex m_conf_mutex;
    uint32_t m_conf_stack_vm_size = 0;
    uint32_t m_conf_warn_timeout = 0;
    uint32_t m_conf_fail_timeout = 0;

    mutable plain_mutex m_queue_mutex;
    struct Queued_Fiber;
    vector<shared_ptr<Queued_Fiber>> m_queue;

    mutable plain_mutex m_sched_mutex;
    weak_ptr<Queued_Fiber> m_sched_self_opt;
    void* m_sched_asan_save;  // private data for address sanitizer
    ::ucontext_t m_sched_outer[1];  // yield target
    long m_sched_wait_ns = 0;

  public:
    // Constructs an empty scheduler.
    Fiber_Scheduler();

  public:
    ASTERIA_NONCOPYABLE_DESTRUCTOR(Fiber_Scheduler);

    // Schedules fibers.
    // This function should be called by the fiber thread repeatedly.
    void
    thread_loop();

    // Reloads configuration from 'main.conf'.
    // If this function fails, an exception is thrown, and there is no effect.
    // This function is thread-safe.
    void
    reload(const Config_File& file);

    // Returns the number of fibers that are being scheduled.
    // This function is thread-safe.
    ROCKET_PURE
    size_t
    count() const noexcept;

    // Inserts a fiber. The scheduler will take ownership of this fiber.
    // This function is thread-safe.
    void
    insert(unique_ptr<Abstract_Fiber>&& fiber);

    // Gets the current fiber if one is being scheduled.
    // This function shall be called from the same thread as `thread_loop()`.
    ROCKET_CONST
    Abstract_Fiber*
    self_opt() const noexcept;

    // Suspends the current fiber until a future becomes satisfied. `self_opt()`
    // must not return a null pointer when this function is called. If no future
    // is specified, this function relinquishes the current time slice, similar
    // to `sched_yield()`. Suspension may not exceed `fiber.fail_timeout` in
    // 'main.conf'.
    void
    checked_yield(const Abstract_Fiber* current, const shared_ptr<Abstract_Future>& futr_opt);
  };

}  // namespace poseidon

#endif
