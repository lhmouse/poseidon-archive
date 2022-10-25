// This file is part of Poseidon.
// Copyleft 2022, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_STATIC_TASK_EXECUTOR_
#define POSEIDON_STATIC_TASK_EXECUTOR_

#include "../fwd.hpp"

namespace poseidon {

// This class buffers tasks and execute them asynchronously.
// Objects of this class are recommended to be static.
class Async_Task_Executor
  {
  private:
    mutable plain_mutex m_queue_mutex;
    condition_variable m_queue_avail;
    vector<weak_ptr<Abstract_Async_Task>> m_queue_buffer;
    size_t m_queue_offset = 0;

  public:
    // Creates an empty task executor.
    explicit
    Async_Task_Executor();

  public:
    ASTERIA_NONCOPYABLE_DESTRUCTOR(Async_Task_Executor);

    // Pops and executes a task.
    // This function should be called by the task thread repeatedly.
    void
    thread_loop();

    // Enqueues a task.
    // If this function fails, an exception is thrown, and there is no effect.
    // This function is thread-safe.
    void
    enqueue(const shared_ptr<Abstract_Async_Task>& task);
  };

}  // namespace poseidon
#endif
