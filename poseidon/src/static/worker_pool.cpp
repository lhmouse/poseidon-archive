// This file is part of Poseidon.
// Copyleft 2020, LH_Mouse. All wrongs reserved.

#include "../precompiled.hpp"
#include "worker_pool.hpp"
#include "main_config.hpp"
#include "../core/abstract_async_job.hpp"
#include "../core/config_file.hpp"
#include "../xutilities.hpp"

namespace poseidon {
namespace {

size_t
do_get_size_config(const Config_File& file, const char* name, long max, size_t def)
  {
    const auto qval = file.get_int64_opt({"worker",name});
    if(!qval)
      return def;

    int64_t rval = ::rocket::clamp(*qval, 1, max);
    if(*qval != rval)
      POSEIDON_LOG_WARN("Config value `worker.poll.$1` truncated to `$2`\n"
                        "[value `$3` out of range]",
                        name, rval, *qval);

    return static_cast<size_t>(rval);
  }

struct Worker
  {
    ::rocket::once_flag init_once;
    ::pthread_t thread;

    mutex queue_mutex;
    condition_variable queue_avail;
    ::std::deque<rcptr<Abstract_Async_Job>> queue;
  };

}  // namespace

POSEIDON_STATIC_CLASS_DEFINE(Worker_Pool)
  {
    // constant data
    ::std::vector<Worker> m_workers;

    static
    void
    do_worker_init_once(Worker* qwrk)
      {
        size_t index = static_cast<size_t>(qwrk - self->m_workers.data());
        auto name = format_string("worker $1", index);
        POSEIDON_LOG_INFO("Creating new worker thread: $1", name);

        mutex::unique_lock lock(qwrk->queue_mutex);
        qwrk->thread = create_daemon_thread<do_worker_thread_loop>(name.c_str(), qwrk);
      }

    static
    void
    do_worker_thread_loop(void* param)
      {
        Worker* const qwrk = static_cast<Worker*>(param);
        rcptr<Abstract_Async_Job> job;

        // Await a job and pop it.
        mutex::unique_lock lock(qwrk->queue_mutex);
        for(;;) {
          job.reset();
          if(qwrk->queue.empty()) {
            // Wait until an element becomes available.
            qwrk->queue_avail.wait(lock);
            continue;
          }

          // Pop it.
          job = ::std::move(qwrk->queue.front());
          qwrk->queue.pop_front();

          if(job.unique() && !job->m_resident.load(::std::memory_order_relaxed)) {
            // Delete this job when no other reference of it exists.
            POSEIDON_LOG_DEBUG("Killed orphan asynchronous job: $1", job);
            continue;
          }

          // Use it.
          POSEIDON_LOG_TRACE("Starting execution of asynchronous job `$1`", job);
          break;
        }
        lock.unlock();

        // Execute the job.
        // See comments in 'abstract_async_job.hpp' for details.
        job->m_state.store(async_state_running, ::std::memory_order_release);
        try {
          job->do_execute();
        }
        catch(exception& stdex) {
          POSEIDON_LOG_WARN("Exception thrown from asynchronous job: $1\n"
                            "[job class `$2`]",
                            stdex.what(), typeid(*job).name());

          // Mark failure.
          job->do_set_exception(::std::current_exception());
        }
        job->m_state.store(async_state_finished, ::std::memory_order_release);
        POSEIDON_LOG_TRACE("Finished execution of asynchronous job `$1`", job);
      }
  };

void
Worker_Pool::
reload()
  {
    // Load worker settings into temporary objects.
    auto file = Main_Config::copy();
    size_t thread_count = do_get_size_config(file, "thread_count", 256, 1);

    // Create the pool without creating threads.
    // Note the pool cannot be resized, so we only have to do this once.
    // No locking is needed.
    if(self->m_workers.empty())
      self->m_workers = ::std::vector<Worker>(thread_count);
  }

size_t
Worker_Pool::
thread_count()
noexcept
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
      POSEIDON_THROW("Null job pointer not valid");

    if(!job.unique())
      POSEIDON_THROW("Job pointer must be unique");

    // Assign the job to a worker.
    if(self->m_workers.empty())
      POSEIDON_THROW("No worker available");

    Worker* const qwrk = ::rocket::get_probing_origin(
                                      self->m_workers.data(),
                                      self->m_workers.data() + self->m_workers.size(),
                                      job->m_key);

    // Perform lazy initialization as necessary.
    qwrk->init_once.call(self->do_worker_init_once, qwrk);

    // Lock the job queue for modification.
    mutex::unique_lock lock(qwrk->queue_mutex);

    // Insert the job.
    qwrk->queue.emplace_back(job);
    job->m_state.store(async_state_pending, ::std::memory_order_release);
    qwrk->queue_avail.notify_one();
    return job;
  }

}  // namespace poseidon
