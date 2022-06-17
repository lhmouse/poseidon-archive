// This file is part of Poseidon.
// Copyleft 2022, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_STATIC_TASK_EXECUTOR_
#define POSEIDON_STATIC_TASK_EXECUTOR_

#include "../fwd.hpp"

namespace poseidon {

// This class buffers tasks and execute them asynchronously.
// Objects of this class are recommended to be static.
class Task_Executor
  {
  private:
    mutable plain_mutex m_queue_mutex;
    condition_variable m_queue_avail;
    struct Queued_Task;
    vector<Queued_Task> m_queue;
    vector<Queued_Task> m_current;

  public:
    // Creates an empty task executor.
    explicit
    Task_Executor();

  public:
    ASTERIA_NONCOPYABLE_DESTRUCTOR(Task_Executor);

    // Pops and executes tasks.
    // This function should be called by the task thread repeatedly.
    void
    thread_loop();

    // Enqueues a task.
    // If this function fails, an exception is thrown, and there is no effect.
    // This function is thread-safe.
    void
    enqueue(const shared_ptr<Abstract_Task>& task);
  };

}  // namespace poseidon

#endif
