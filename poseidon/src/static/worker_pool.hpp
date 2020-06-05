// This file is part of Poseidon.
// Copyleft 2020, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_STATIC_WORKER_POOL_HPP_
#define POSEIDON_STATIC_WORKER_POOL_HPP_

#include "../fwd.hpp"

namespace poseidon {

class Worker_Pool
  {
    POSEIDON_STATIC_CLASS_DECLARE(Worker_Pool);

  private:
    static inline
    void
    do_thread_loop(void* param);

  public:
    // Initializes the thread pool.
    // Note this doesn't create threads. They are created only when needed.
    // The pool cannot be resized.
    static
    void
    start();

    // Retrieves the maximum number of worker threads.
    // This function is thread-safe.
    ROCKET_PURE_FUNCTION static
    size_t
    thread_count()
    noexcept;

    // Inserts an asynchronous job.
    // Functions with the same key will be delivered to the same worker thread.
    // If this function fails, an exception is thrown, and there is no effect.
    // This function is thread-safe.
    static
    rcptr<Abstract_Async_Job>
    insert(uptr<Abstract_Async_Job>&& ufunc);
  };

}  // namespace poseidon

#endif
