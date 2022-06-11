// This file is part of Poseidon.
// Copyleft 2022, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_STATIC_FIBER_SCHEDULER_
#define POSEIDON_STATIC_FIBER_SCHEDULER_

#include "../fwd.hpp"
#include "../core/config_file.hpp"

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

    mutable plain_mutex m_sched_mutex;
    vector<unique_ptr<Abstract_Fiber>> m_sched_queue;

    mutable plain_mutex m_exec_mutex;
    Abstract_Fiber* m_exec_self_opt = nullptr;

  public:
    // Constructs an empty scheduler.
    Fiber_Scheduler();

  public:
    ASTERIA_NONCOPYABLE_DESTRUCTOR(Fiber_Scheduler);

    // Schedules fibers.
    // This function should be called by the fiber thread repeatedly.
    void
    thread_loop();

    // Gets the current fiber if one is being scheduled.
    // This function shall be called from the same thread as `thread_loop()`.
    ROCKET_CONST
    Abstract_Fiber*
    self_opt() const noexcept
      {
        return this->m_exec_self_opt;
      }

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
  };

}  // namespace poseidon

#endif
