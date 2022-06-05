// This file is part of Poseidon.
// Copyleft 2020, LH_Mouse. All wrongs reserved.

#include "../precompiled.ipp"
#include "task_executor_pool.hpp"
#include "async_logger.hpp"
#include "main_config.hpp"
#include "../core/abstract_task.hpp"
#include "../core/config_file.hpp"
#include "../utils.hpp"
#include <signal.h>

namespace poseidon {

struct Task_Executor_Pool::Executor
  {
    once_flag m_init_once;
    ::pthread_t m_thread;

    mutable simple_mutex m_queue_mutex;
    condition_variable m_queue_avail;
    ::std::deque<rcptr<Abstract_Task>> m_queue;

    Executor() = default;
    Executor(const Executor&) = delete;
    Executor& operator=(const Executor&) = delete;
  };

POSEIDON_STATIC_CLASS_DEFINE(Task_Executor_Pool)
  {
    // dynamic data
    ::std::vector<Executor> m_executors;

    [[noreturn]] static
    void*
    do_thread_procedure(void* param)
      {
        // Set thread information. Errors are ignored.
        ::sigset_t sigset;
        ::sigemptyset(&sigset);
        ::sigaddset(&sigset, SIGINT);
        ::sigaddset(&sigset, SIGTERM);
        ::sigaddset(&sigset, SIGHUP);
        ::sigaddset(&sigset, SIGALRM);
        ::pthread_sigmask(SIG_BLOCK, &sigset, nullptr);

        int oldst;
        ::pthread_setcancelstate(PTHREAD_CANCEL_DISABLE, &oldst);

        char name[32];
        Executor* executor = (Executor*) param;
        ::sprintf(name, "executor %u", (unsigned) (executor - self->m_executors.data() + 1));
        ::pthread_setname_np(::pthread_self(), name);

        // Enter an infinite loop.
        for(;;)
          try {
            self->do_thread_loop(executor);
          }
          catch(exception& stdex) {
            POSEIDON_LOG_FATAL(
                "Caught an exception from executor thread loop: $1\n"
                "[exception class `$2`]\n",
                stdex.what(), typeid(stdex).name());
          }
      }

    static
    void
    do_thread_loop(Executor* executor)
      {
        // Await a task and pop it.
        simple_mutex::unique_lock lock(executor->m_queue_mutex);
        executor->m_queue_avail.wait(lock, [&] { return executor->m_queue.size();  });

        auto task = ::std::move(executor->m_queue.front());
        executor->m_queue.pop_front();
        lock.unlock();

        if(task->m_zombie.load()) {
          POSEIDON_LOG_DEBUG("Shut down asynchronous task: $1 ($2)", task, typeid(*task));
          return;
        }
        else if(task.unique() && !task->m_resident.load()) {
          POSEIDON_LOG_DEBUG("Killed orphan asynchronous task: $1 ($2)", task, typeid(*task));
          return;
        }

        // Execute the task.
        ROCKET_ASSERT(task->state() == async_state_pending);
        task->m_state.store(async_state_running);
        POSEIDON_LOG_TRACE("Starting execution of asynchronous task `$1`", task);

        try {
          task->do_abstract_task_execute();
        }
        catch(exception& stdex) {
          POSEIDON_LOG_WARN(
              "$1\n"
              "[thrown from an asynchronous task of class `$2`]",
              stdex, typeid(*task));
        }

        ROCKET_ASSERT(task->state() == async_state_running);
        task->m_state.store(async_state_finished);
        POSEIDON_LOG_TRACE("Finished execution of asynchronous task `$1`", task);
      }
  };

void
Task_Executor_Pool::
reload()
  {
    // Load executor settings into temporary objects.
    const auto file = Main_Config::copy();
    uint32_t thread_count = 1;

    auto qint = file.get_int64_opt({"task","thread_count"});
    if(qint)
      thread_count = clamp_cast<uint32_t>(*qint, 1, 127);

    // Create the pool without creating threads.
    // Note the pool cannot be resized, so we only have to do this once.
    // No locking is needed.
    if(self->m_executors.empty())
      self->m_executors = ::std::vector<Executor>(thread_count);
  }

size_t
Task_Executor_Pool::
thread_count() noexcept
  {
    return self->m_executors.size();
  }

rcptr<Abstract_Task>
Task_Executor_Pool::
insert(uptr<Abstract_Task>&& utask)
  {
    // Take ownership of `utask`.
    rcptr<Abstract_Task> task(utask.release());
    if(!task)
      POSEIDON_THROW("null task pointer not valid");

    if(!task.unique())
      POSEIDON_THROW("task pointer must be unique");

    // Assign the task to a executor.
    size_t nexecutors = self->m_executors.size();
    if(nexecutors == 0)
      POSEIDON_THROW("no executor available");

    uint64_t index = (uint32_t) (task->m_key * 0x9E3779B9);
    index *= nexecutors;
    index >>= 32;
    ROCKET_ASSERT(index < nexecutors);
    const auto executor = self->m_executors.data() + index;

    // Perform initialization.
    executor->m_init_once.call(
      [executor] {
        POSEIDON_LOG_INFO("Initializing executor thread...");
        simple_mutex::unique_lock lock(executor->m_queue_mutex);

        // Create the thread. Note it is never joined or detached.
        int err = ::pthread_create(&(executor->m_thread), nullptr, self->do_thread_procedure, executor);
        if(err != 0) ::std::terminate();
      });

    // Perform some initialization. No locking is needed here.
    task->m_state.store(async_state_pending);

    // Insert the task.
    simple_mutex::unique_lock lock(executor->m_queue_mutex);
    executor->m_queue.emplace_back(task);
    executor->m_queue_avail.notify_one();
    return task;
  }

}  // namespace poseidon
