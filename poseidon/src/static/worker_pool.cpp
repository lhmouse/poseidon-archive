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
    ::rocket::once_flag m_init_once;
    ::pthread_t m_thread;

    mutex m_queue_mutex;
    condition_variable m_queue_avail;
    ::std::deque<rcptr<Abstract_Async_Job>> m_queue;
  };

}  // namespace

POSEIDON_STATIC_CLASS_DEFINE(Worker_Pool)
  {
    // constant data
    ::std::vector<Worker> m_workers;
  };

void
Worker_Pool::
do_worker_thread_loop(void* param)
  {
    // Await a function and pop it.
    auto& worker = *(static_cast<Worker*>(param));
    mutex::unique_lock lock(worker.m_queue_mutex);
    while(worker.m_queue.empty())
      worker.m_queue_avail.wait(lock);

    const auto func = ::std::move(worker.m_queue.front());
    worker.m_queue.pop_front();
    lock.unlock();

    // Execute the function.
    // See comments in 'abstract_async_job.hpp' for details.
    func->m_state.store(async_state_running, ::std::memory_order_relaxed);
    try {
      func->do_execute();
    }
    catch(exception& stdex) {
      POSEIDON_LOG_WARN("Exception thrown from asynchronous job: $1\n"
                        "[function class `$2`]",
                        stdex.what(), typeid(*func).name());

      func->do_set_exception(::std::current_exception());
    }
    func->m_state.store(async_state_finished, ::std::memory_order_release);
  }

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
insert(uptr<Abstract_Async_Job>&& ufunc)
  {
    // Take ownership of `ufunc`.
    rcptr<Abstract_Async_Job> func(ufunc.release());
    if(!func)
      POSEIDON_THROW("null function pointer not valid");

    if(!func.unique())
      POSEIDON_THROW("function pointer must be unique");

    // Locate a worker and lock it.
    auto bptr = self->m_workers.data();
    auto eptr = bptr + self->m_workers.size();
    if(bptr == eptr)
      POSEIDON_THROW("no worker available");

    eptr = ::rocket::get_probing_origin(bptr, eptr, func->m_key);
    auto& worker = *eptr;

    mutex::unique_lock lock(worker.m_queue_mutex);

    // Create the worker on demand.
    ::rocket::call_once(worker.m_init_once,
      [&] {
        ptrdiff_t index = eptr - bptr;
        auto name = format_string("worker $1", index);
        POSEIDON_LOG_INFO("Creating new worker thread: $1", name);
        worker.m_thread = create_daemon_thread<do_worker_thread_loop>(
                                                        name.c_str(), eptr);
      });

    // Insert the function.
    worker.m_queue.emplace_back(func);
    func->m_state.store(async_state_pending, ::std::memory_order_relaxed);
    worker.m_queue_avail.notify_one();
    return func;
  }

}  // namespace poseidon
