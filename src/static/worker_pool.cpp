// This file is part of Poseidon.
// Copyleft 2020, LH_Mouse. All wrongs reserved.

#include "../precompiled.ipp"
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

    Worker() = default;
    Worker(const Worker&) = delete;
    Worker& operator=(const Worker&) = delete;
  };

}  // namespace

POSEIDON_STATIC_CLASS_DEFINE(Worker_Pool)
  {
    // dynamic data
    ::rocket::cow_vector<Worker> m_workers;

    static
    void
    do_worker_init_once(Worker* qwrk)
      {
        size_t index = static_cast<size_t>(qwrk - self->m_workers.data());
        auto name = format_string("worker $1", index);
        POSEIDON_LOG_INFO("Creating new worker thread: $1", name);

        simple_mutex::unique_lock lock(qwrk->queue_mutex);
        qwrk->thread = create_daemon_thread<do_worker_thread_loop>(name.c_str(), qwrk);
      }

    static
    void
    do_worker_thread_loop(void* param)
      {
        const auto qwrk = static_cast<Worker*>(param);

        // Await a job and pop it.
        simple_mutex::unique_lock lock(qwrk->queue_mutex);
        while(qwrk->queue.empty())
          qwrk->queue_avail.wait(lock);

        auto job = ::std::move(qwrk->queue.front());
        if(job->m_zombie.load()) {
          // Delete this job asynchronously.
          POSEIDON_LOG_DEBUG("Shut down asynchronous job: $1", job);
          qwrk->queue.pop_front();
          return;
        }
        else if(job.unique() && !job->m_resident.load()) {
          // Delete this job when no other reference of it exists.
          POSEIDON_LOG_DEBUG("Killed orphan asynchronous job: $1", job);
          qwrk->queue.pop_front();
          return;
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
          POSEIDON_LOG_WARN(
              "$1\n[thrown from an asynchronous job of class `$2`]",
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

    auto qwrk = self->m_workers.mut_data();
    qwrk = ::rocket::get_probing_origin(qwrk, qwrk + nworkers, job->m_key);
    qwrk->init_once.call(self->do_worker_init_once, qwrk);

    // Perform some initialization. No locking is needed here.
    job->m_state.store(async_state_pending);

    // Insert the job.
    simple_mutex::unique_lock lock(qwrk->queue_mutex);
    qwrk->queue.emplace_back(job);
    qwrk->queue_avail.notify_one();
    return job;
  }

}  // namespace poseidon
