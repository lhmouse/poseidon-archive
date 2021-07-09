// This file is part of Poseidon.
// Copyleft 2020, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_STATIC_WORKER_POOL_HPP_
#define POSEIDON_STATIC_WORKER_POOL_HPP_

#include "../fwd.hpp"

namespace poseidon {

class Worker_Pool
  {
    POSEIDON_STATIC_CLASS_DECLARE(Worker_Pool);

  public:
    // Reloads settings from main config.
    // If this function fails, an exception is thrown, and there is no effect.
    // Note that the number of threads is set upon the first call and cannot be
    // changed thereafter.
    // This function is thread-safe.
    static void
    reload();

    // Retrieves the maximum number of worker threads.
    // This function is thread-safe.
    ROCKET_PURE static size_t
    thread_count() noexcept;

    // Inserts an asynchronous job.
    // Functions with the same key will be delivered to the same worker thread.
    // The worker pool holds a reference-counted pointer to the job. If the job has no
    // other references elsewhere and has not started execution, it is deleted without
    // being executed at all.
    // If this function fails, an exception is thrown, and there is no effect.
    // This function is thread-safe.
    static rcptr<Abstract_Async_Job>
    insert(uptr<Abstract_Async_Job>&& ujob);
  };

}  // namespace poseidon

#endif
