// This file is part of Poseidon.
// Copyleft 2022, LH_Mouse. All wrongs reserved.

#include "../precompiled.ipp"
#include "task_executor.hpp"
#include "async_logger.hpp"
#include "../core/abstract_task.hpp"
#include "../utils.hpp"

namespace poseidon {
namespace {

struct Queued_Task
  {
    weak_ptr<Abstract_Task> task;
  };

}  // namespace

POSEIDON_HIDDEN_STRUCT(Task_Executor, Queued_Task);

Task_Executor::
Task_Executor()
  {
  }

Task_Executor::
~Task_Executor()
  {
  }

void
Task_Executor::
thread_loop()
  {
    // Await a task.
    plain_mutex::unique_lock lock(this->m_queue_mutex);
    while(this->m_queue.empty() && this->m_current.empty())
      this->m_queue_avail.wait(lock);

    if(this->m_current.empty())
      this->m_current.swap(this->m_queue);

    auto elem = ::std::move(this->m_current.back());
    lock.unlock();

    auto task = elem.task.lock();
    if(!task)
      return;

    // Execute it.
    // Exceptions are ignored.
    POSEIDON_LOG_TRACE(("Executing task `$1` (class `$2`)"), task, typeid(*task));
    task->m_state.store(async_state_running);

    try {
      task->do_abstract_task_on_execution();
    }
    catch(exception& stdex) {
      POSEIDON_LOG_ERROR((
          "Unhandled exception thrown from asynchronous task: $1",
          "[task class `$2`]"),
          stdex, typeid(*task));
    }

    ROCKET_ASSERT(task->m_state.load() == async_state_running);
    task->m_state.store(async_state_finished);
  }

void
Task_Executor::
enqueue(const shared_ptr<Abstract_Task>& task)
  {
    // Validate arguments.
    if(!task)
      POSEIDON_THROW(("Null task pointer not valid"));

    // Initialize the element to insert.
    Queued_Task elem;
    elem.task = task;

    // Insert the task.
    plain_mutex::unique_lock lock(this->m_queue_mutex);
    this->m_queue.emplace_back(::std::move(elem));
    this->m_queue_avail.notify_one();

  }

}  // namespace poseidon
