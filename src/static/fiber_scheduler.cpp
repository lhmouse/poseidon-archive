// This file is part of Poseidon.
// Copyleft 2022, LH_Mouse. All wrongs reserved.

#include "../precompiled.ipp"
#include "fiber_scheduler.hpp"
#include "async_logger.hpp"
#include "../core/abstract_fiber.hpp"
#include "../utils.hpp"
#include <time.h>

namespace poseidon {
namespace {

}  // namespace

Fiber_Scheduler::
Fiber_Scheduler()
  {
  }

Fiber_Scheduler::
~Fiber_Scheduler()
  {
  }

void
Fiber_Scheduler::
thread_loop()
  {
    POSEIDON_LOG_INFO(("fiber scheduler running"));
    ::sleep(1);
  }

void
Fiber_Scheduler::
reload(const Config_File& file)
  {
  }

size_t
Fiber_Scheduler::
count() const noexcept
  {
    plain_mutex::unique_lock lock(this->m_sched_mutex);
    return this->m_sched_queue.size();
  }

void
Fiber_Scheduler::
insert(unique_ptr<Abstract_Fiber>&& fiber)
  {
  }

}  // namespace poseidon
