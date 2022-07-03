// This file is part of Poseidon.
// Copyleft 2022, LH_Mouse. All wrongs reserved.

#include "../precompiled.ipp"
#include "task_executor.hpp"
#include "async_logger.hpp"
#include "../core/abstract_task.hpp"
#include "../utils.hpp"

namespace poseidon {

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
    plain_mutex::unique_lock lock(this->m_queue_mutex);
    while(this->m_queue_offset == this->m_queue_buffer.size())
      this->m_queue_avail.wait(lock);

    auto task = this->m_queue_buffer[this->m_queue_offset].lock();
    this->m_queue_offset ++;
    if(this->m_queue_offset == this->m_queue_buffer.size()) {
      POSEIDON_LOG_TRACE(("Clearing task queue: size = $1"), this->m_queue_buffer.size());
      this->m_queue_buffer.clear();
      this->m_queue_offset = 0;
    }
    lock.unlock();

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
    if(!task)
      POSEIDON_THROW(("Null task pointer not valid"));

    // Insert the task.
    plain_mutex::unique_lock lock(this->m_queue_mutex);
    this->m_queue_buffer.emplace_back(task);
    this->m_queue_avail.notify_one();

  }

}  // namespace poseidon
