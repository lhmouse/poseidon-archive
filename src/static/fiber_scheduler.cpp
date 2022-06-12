// This file is part of Poseidon.
// Copyleft 2022, LH_Mouse. All wrongs reserved.

#include "../precompiled.ipp"
#include "fiber_scheduler.hpp"
#include "async_logger.hpp"
#include "../core/abstract_fiber.hpp"
#include "../utils.hpp"
#include <time.h>  // clock_gettime()
#include <sys/resource.h>  // getrlimit()

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
    // Parse new configuration. Default ones are defined here.
    int64_t stack_vm_size = 0;
    int64_t warn_timeout = 15;
    int64_t fail_timeout = 300;

    // Read the stack size from configuration.
    auto value = file.query("fiber", "stack_vm_size");
    if(value.is_integer())
      stack_vm_size = value.as_integer();
    else if(!value.is_null())
      POSEIDON_LOG_WARN((
          "Ignoring `fiber.stack_vm_size`: expecting an `integer`, got `$1`",
          "[in configuration file '$2']"),
          value, file.path());

    // If no value or zero is specified, use the system's stack size.
    if(stack_vm_size == 0) {
      ::rlimit rlim;
      if(::getrlimit(RLIMIT_STACK, &rlim) != 0)
        POSEIDON_THROW((
            "Could not get system stack size",
            "[`getrlimit()` failed: $1]"),
            format_errno());

      POSEIDON_LOG_DEBUG(("System stack size: $1 bytes"), rlim.rlim_cur);
      stack_vm_size = (int64_t) rlim.rlim_cur;
    }

    if((stack_vm_size < 0x10000) || (stack_vm_size > 0x7FFF0000))
      POSEIDON_THROW((
          "`fiber.stack_vm_size` value `$1` out of range",
          "[in configuration file '$2']"),
          stack_vm_size, file.path());

    if(stack_vm_size & 0xFFFF)
      POSEIDON_THROW((
          "`fiber.stack_vm_size` value `$1` not a multiple of 64KiB",
          "[in configuration file '$2']"),
          stack_vm_size, file.path());

    // Read scheduler timeouts inseconds.
    value = file.query("fiber", "warn_timeout");
    if(value.is_integer())
      warn_timeout = value.as_integer();
    else if(!value.is_null())
      POSEIDON_LOG_WARN((
          "Ignoring `fiber.warn_timeout`: expecting an `integer`, got `$1`",
          "[in configuration file '$2']"),
          value, file.path());

    if((warn_timeout < 0) || (warn_timeout > 86400))
      POSEIDON_THROW((
          "`fiber.warn_timeout` value `$1` out of range",
          "[in configuration file '$2']"),
          warn_timeout, file.path());

    value = file.query("fiber", "fail_timeout");
    if(value.is_integer())
      fail_timeout = value.as_integer();
    else if(!value.is_null())
      POSEIDON_LOG_WARN((
          "Ignoring `fiber.fail_timeout`: expecting an `integer`, got `$1`",
          "[in configuration file '$2']"),
          value, file.path());

    if((fail_timeout < 0) || (fail_timeout > 86400))
      POSEIDON_THROW((
          "`fiber.fail_timeout` value `$1` out of range",
          "[in configuration file '$2']"),
          fail_timeout, file.path());

    // Set up new data.
    plain_mutex::unique_lock lock(this->m_conf_mutex);
    this->m_conf_stack_vm_size = (uint32_t) stack_vm_size;
    this->m_conf_warn_timeout = (uint32_t) warn_timeout;
    this->m_conf_fail_timeout = (uint32_t) fail_timeout;
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
    // Validate arguments.
    if(!fiber)
      POSEIDON_THROW(("Null fiber pointer not valid"));

    // Insert the fiber.
    plain_mutex::unique_lock lock(this->m_sched_mutex);
    fiber->m_sched_timeout = 0;  // starts immediately
    fiber->m_sched_yield_since = 0;
    this->m_sched_queue.emplace_back(::std::move(fiber));
  }

void
Fiber_Scheduler::
yield(const shared_ptr<Abstract_Future>& futr_opt)
  {
  }

}  // namespace poseidon
