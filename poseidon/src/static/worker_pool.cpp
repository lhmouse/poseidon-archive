// This file is part of Poseidon.
// Copyleft 2020, LH_Mouse. All wrongs reserved.

#include "../precompiled.hpp"
#include "worker_pool.hpp"
#include "main_config.hpp"
#include "../core/abstract_async_job.hpp"
#include "../core/config_file.hpp"
#include "../utils.hpp"

namespace poseidon {
namespace {

struct Worker
  {
    once_flag init_once;
    ::pthread_t thread;

    mutable simple_mutex queue_mutex;
    condition_variable queue_avail;
    ::std::deque<rcptr<Abstract_Async_Job>> queue;

    // Declare dummy constructors because this structure will be placed in
    // a vector, which is only resized when it is empty so these will never
    // be called eventually.
    Worker()
      = default;

    Worker(const Worker&)
      { ROCKET_ASSERT(false);  }

    Worker&
    operator=(const Worker&)
      { ROCKET_ASSERT(false);  }
  };

}  // namespace

POSEIDON_STATIC_CLASS_DEFINE(Worker_Pool)
  {
    // dynamic data
    ::std::vector<Worker> m_workers;

    static void
    do_worker_init_once(Worker* qworker)
      {
        size_t index = static_cast<size_t>(qworker - self->m_workers.data());
        auto name = format_string("worker $1", index);
        POSEIDON_LOG_INFO("Creating new worker thread: $1", name);

        simple_mutex::unique_lock lock(qworker->queue_mutex);
        qworker->thread = create_daemon_thread<do_worker_thread_loop>(
                                                   name.c_str(), qworker);
      }

    static void
    do_worker_thread_loop(void* param)
      {
        auto qworker = static_cast<Worker*>(param);
        rcptr<Abstract_Async_Job> job;

        // Await a job and pop it.
        simple_mutex::unique_lock lock(qworker->queue_mutex);
        for(;;) {
          job.reset();
          if(qworker->queue.empty()) {
            // Wait until an element becomes available.
            qworker->queue_avail.wait(lock);
            continue;
          }

          // Pop it.
          job = ::std::move(qworker->queue.front());
          qworker->queue.pop_front();

          if(job->m_zombie.load()) {
            // Delete this job asynchronously.
            POSEIDON_LOG_DEBUG("Shut down asynchronous job: $1", job);
            continue;
          }

          if(job.unique() && !job->m_resident.load()) {
            // Delete this job when no other reference of it exists.
            POSEIDON_LOG_DEBUG("Killed orphan asynchronous job: $1", job);
            continue;
          }

          // Use it.
          break;
        }
        lock.unlock();

        // Execute the job.
        ROCKET_ASSERT(job->state() == async_state_pending);
        job->m_state.store(async_state_running);
        POSEIDON_LOG_TRACE("Starting execution of asynchronous job `$1`", job);

        try {
          job->do_execute();
        }
        catch(exception& stdex) {
          POSEIDON_LOG_WARN("$1\n[thrown from an asynchronous job of class `$2`]",
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
      self->m_workers.resize(thread_count);
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

    auto qworker = ::rocket::get_probing_origin(self->m_workers.data(),
                           self->m_workers.data() + nworkers, job->m_key);

    qworker->init_once.call(self->do_worker_init_once, qworker);

    // Perform some initialization. No locking is needed here.
    job->m_state.store(async_state_pending);

    // Insert the job.
    simple_mutex::unique_lock lock(qworker->queue_mutex);
    qworker->queue.emplace_back(job);
    qworker->queue_avail.notify_one();
    return job;
  }

}  // namespace poseidon
