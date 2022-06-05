// This file is part of Poseidon.
// Copyleft 2020, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_STATIC_TASK_EXECUTOR_POOL_
#define POSEIDON_STATIC_TASK_EXECUTOR_POOL_

#include "../fwd.hpp"

namespace poseidon {

class Task_Executor_Pool
  {
    POSEIDON_STATIC_CLASS_DECLARE(Task_Executor_Pool);
    struct __attribute__((__visibility__("hidden"))) Executor;

  public:
    // Reloads settings from main config.
    // If this function fails, an exception is thrown, and there is no effect.
    // Note that the number of threads is set upon the first call and cannot be
    // changed thereafter.
    // This function is thread-safe.
    static
    void
    reload();

    // Retrieves the maximum number of executor threads.
    // This function is thread-safe.
    ROCKET_PURE static
    size_t
    thread_count() noexcept;

    // Inserts an asynchronous task.
    // Functions with the same key will be delivered to the same executor thread.
    // The executor pool holds a reference-counted pointer to the task. If the task
    // has no other references elsewhere and has not started execution, it is
    // deleted without being executed at all.
    // If this function fails, an exception is thrown, and there is no effect.
    // This function is thread-safe.
    static
    rcptr<Abstract_Task>
    insert(uptr<Abstract_Task>&& utask);
  };

}  // namespace poseidon

#endif
