// This file is part of Poseidon.
// Copyleft 2020, LH_Mouse. All wrongs reserved.

#include "../precompiled.ipp"
#include "worker_pool.hpp"
#include "async_logger.hpp"
#include "main_config.hpp"
#include "../core/abstract_async_job.hpp"
#include "../core/config_file.hpp"
#include "../utils.hpp"
#include <signal.h>

namespace poseidon {

struct Worker_Pool::Worker
  {
    once_flag m_init_once;
    ::pthread_t m_thread;

    mutable simple_mutex m_queue_mutex;
    condition_variable m_queue_avail;
    ::std::deque<rcptr<Abstract_Async_Job>> m_queue;

    Worker() = default;
    Worker(const Worker&) = delete;
    Worker& operator=(const Worker&) = delete;
  };

POSEIDON_STATIC_CLASS_DEFINE(Worker_Pool)
  {
    // dynamic data
    ::std::vector<Worker> m_workers;

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
        Worker* worker = (Worker*) param;
        ::sprintf(name, "worker %u", (unsigned) (worker - self->m_workers.data() + 1));
        ::pthread_setname_np(::pthread_self(), name);

        // Enter an infinite loop.
        for(;;)
          try {
            self->do_thread_loop(worker);
          }
          catch(exception& stdex) {
            POSEIDON_LOG_FATAL(
                "Caught an exception from worker thread loop: $1\n"
                "[exception class `$2`]\n",
                stdex.what(), typeid(stdex).name());
          }
      }

    static
    void
    do_thread_loop(Worker* worker)
      {
        // Await a job and pop it.
        simple_mutex::unique_lock lock(worker->m_queue_mutex);
        worker->m_queue_avail.wait(lock, [&] { return worker->m_queue.size();  });

        auto job = ::std::move(worker->m_queue.front());
        worker->m_queue.pop_front();
        lock.unlock();

        if(job->m_zombie.load()) {
          POSEIDON_LOG_DEBUG("Shut down asynchronous job: $1 ($2)", job, typeid(*job));
          return;
        }
        else if(job.unique() && !job->m_resident.load()) {
          POSEIDON_LOG_DEBUG("Killed orphan asynchronous job: $1 ($2)", job, typeid(*job));
          return;
        }

        try {
          // Execute the job.
          ROCKET_ASSERT(job->state() == async_state_pending);
          job->m_state.store(async_state_running);
          POSEIDON_LOG_TRACE("Starting execution of asynchronous job `$1`", job);
          job->do_execute();
        }
        catch(exception& stdex) {
          POSEIDON_LOG_WARN(
              "$1\n"
              "[thrown from an asynchronous job of class `$2`]",
              stdex, typeid(*job));
        }

        ROCKET_ASSERT(job->state() == async_state_running);
        job->m_state.store(async_state_finished);
        POSEIDON_LOG_TRACE("Finished execution of asynchronous job `$1`", job);
      }
  };

void
Worker_Pool::
reload()
  {
    // Load worker settings into temporary objects.
    const auto file = Main_Config::copy();
    uint32_t thread_count = 1;

    auto qint = file.get_int64_opt({"worker","thread_count"});
    if(qint)
      thread_count = clamp_cast<uint32_t>(*qint, 1, 127);

    // Create the pool without creating threads.
    // Note the pool cannot be resized, so we only have to do this once.
    // No locking is needed.
    if(self->m_workers.empty())
      self->m_workers = ::std::vector<Worker>(thread_count);
  }

size_t
Worker_Pool::
thread_count() noexcept
  {
    return self->m_workers.size();
  }

rcptr<Abstract_Async_Job>
Worker_Pool::
insert(uptr<Abstract_Async_Job>&& ujob)
  {
    // Take ownership of `ujob`.
    rcptr<Abstract_Async_Job> job(ujob.release());
    if(!job)
      POSEIDON_THROW("null job pointer not valid");

    if(!job.unique())
      POSEIDON_THROW("job pointer must be unique");

    // Assign the job to a worker.
    size_t nworkers = self->m_workers.size();
    if(nworkers == 0)
      POSEIDON_THROW("no worker available");

    uint64_t index = (uint32_t) (job->m_key * 0x9E3779B9);
    index *= nworkers;
    index >>= 32;
    ROCKET_ASSERT(index < nworkers);
    const auto worker = self->m_workers.data() + index;

    // Perform initialization.
    worker->m_init_once.call(
      [worker] {
        POSEIDON_LOG_INFO("Initializing worker thread...");
        simple_mutex::unique_lock lock(worker->m_queue_mutex);

        // Create the thread. Note it is never joined or detached.
        int err = ::pthread_create(&(worker->m_thread), nullptr, self->do_thread_procedure, worker);
        if(err != 0) ::std::terminate();
      });

    // Perform some initialization. No locking is needed here.
    job->m_state.store(async_state_pending);

    // Insert the job.
    simple_mutex::unique_lock lock(worker->m_queue_mutex);
    worker->m_queue.emplace_back(job);
    worker->m_queue_avail.notify_one();
    return job;
  }

}  // namespace poseidon
