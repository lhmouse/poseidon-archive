// This file is part of Poseidon.
// Copyleft 2022, LH_Mouse. All wrongs reserved.

#include "../precompiled.ipp"
#include "fiber_scheduler.hpp"
#include "async_logger.hpp"
#include "../core/config_file.hpp"
#include "../fiber/abstract_fiber.hpp"
#include "../fiber/abstract_future.hpp"
#include "../utils.hpp"
#include <time.h>  // clock_gettime()
#include <sys/resource.h>  // getrlimit()
#include <sys/mman.h>  // mmap(), munmap()

#ifdef POSEIDON_ENABLE_ADDRESS_SANITIZER
extern "C" void __sanitizer_start_switch_fiber(void**, const void*, size_t) noexcept;
extern "C" void __sanitizer_finish_switch_fiber(void*, const void**, size_t*) noexcept;
#endif  // POSEIDON_ENABLE_ADDRESS_SANITIZER

namespace poseidon {
namespace {

plain_mutex s_stack_pool_mutex;
::stack_t s_stack_pool;

::stack_t
do_alloc_stack(size_t stack_vm_size)
  {
    if(stack_vm_size > 0x7FFF0000)
      POSEIDON_THROW(("Invalid stack size: $1"), stack_vm_size);

    for(;;) {
      // Try popping a cached one.
      plain_mutex::unique_lock lock(s_stack_pool_mutex);
      ::stack_t ss = s_stack_pool;
      if(!ss.ss_sp)
        break;

      ::memcpy(&s_stack_pool, ss.ss_sp, sizeof(ss));
      lock.unlock();

      // If it is large enough, return it.
      if(ss.ss_size >= stack_vm_size)
        return ss;

      // Otherwise, free it and try the next one.
      do_free_stack(ss);
    }

    // Allocate a new stack.
    ::stack_t ss;
    ss.ss_size = stack_vm_size;
    ss.ss_sp = ::mmap(nullptr, ss.ss_size + 0x2000, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_STACK, -1, 0);
    if(ss.ss_sp == (void*) -1)
      POSEIDON_THROW((
          "Could not allocate fiber stack memory of size `$2`",
          "[`mmap()` failed: $1]"),
          format_errno(), ss.ss_size);

    // Adjust the pointer to writable memory, and make it so.
    ss.ss_sp = (char*) ss.ss_sp + 0x1000;
    ::mprotect(ss.ss_sp, ss.ss_size, PROT_READ | PROT_WRITE);
    return ss;
  }

void
do_free_stack(::stack_t ss) noexcept
  {
    if(!ss.ss_sp)
      return;

    if(::munmap((char*) ss.ss_sp - 0x1000, ss.ss_size + 0x2000) != 0)
      POSEIDON_LOG_FATAL((
          "Failed to unmap fiber stack memory `$2` of size `$3`",
          "[`munmap()` failed: $1]"),
          format_errno(), ss.ss_sp, ss.ss_size);
  }

void
do_pool_stack(::stack_t ss) noexcept
  {
    if(!ss.ss_sp)
      return;

    // Prepend the stack to the list.
    plain_mutex::unique_lock lock(s_stack_pool_mutex);
    ::memcpy(ss.ss_sp, &s_stack_pool, sizeof(ss));
    s_stack_pool = ss;
  }

struct Queued_Fiber
  {
    unique_ptr<Abstract_Fiber> fiber;
    atomic_relaxed<int64_t> async_time;  // this might get modified at any time

    weak_ptr<Abstract_Future> futr_opt;
    int64_t yield_time;
    int64_t check_time;
    int64_t fail_timeout_override;
    ::ucontext_t sched_inner[1];
  };

struct Fiber_Comparator
  {
    // We have to build a minheap here.
    bool
    operator()(const shared_ptr<Queued_Fiber>& lhs, const shared_ptr<Queued_Fiber>& rhs) noexcept
      { return lhs->check_time > rhs->check_time;  }

    bool
    operator()(const shared_ptr<Queued_Fiber>& lhs, int64_t rhs) noexcept
      { return lhs->check_time > rhs;  }

    bool
    operator()(int64_t lhs, const shared_ptr<Queued_Fiber>& rhs) noexcept
      { return lhs > rhs->check_time;  }
  }
  constexpr fiber_comparator;

inline
void
do_start_switch_fiber(void*& save, const ::ucontext_t* uctx) noexcept
  {
#ifdef POSEIDON_ENABLE_ADDRESS_SANITIZER
    ::__sanitizer_start_switch_fiber(&save, uctx->uc_stack.ss_sp, uctx->uc_stack.ss_size);
#endif
    (void) save, (void) uctx;
  }

inline
void
do_finish_switch_fiber(void* save) noexcept
  {
#ifdef POSEIDON_ENABLE_ADDRESS_SANITIZER
    ::__sanitizer_finish_switch_fiber(save, nullptr, nullptr);
#endif
    (void) save;
  }

}  // namespace

POSEIDON_HIDDEN_STRUCT(Fiber_Scheduler, Queued_Fiber);

Fiber_Scheduler::
Fiber_Scheduler()
  {
  }

Fiber_Scheduler::
~Fiber_Scheduler()
  {
  }

int64_t
Fiber_Scheduler::
clock() noexcept
  {
    ::timespec ts;
    ::clock_gettime(CLOCK_MONOTONIC_COARSE, &ts);
    return ts.tv_sec * 1000000000LL + ts.tv_nsec;
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

    if(stack_vm_size == 0) {
      // If no value or zero is specified, use the system's stack size.
      ::rlimit rlim;
      if(::getrlimit(RLIMIT_STACK, &rlim) != 0)
        POSEIDON_THROW((
            "Could not get system stack size",
            "[`getrlimit()` failed: $1]"),
            format_errno());

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

void
Fiber_Scheduler::
thread_loop()
  {
    plain_mutex::unique_lock lock(this->m_conf_mutex);
    const size_t stack_vm_size = this->m_conf_stack_vm_size;
    const int64_t warn_timeout = this->m_conf_warn_timeout * 1000000000LL;
    const int64_t fail_timeout = this->m_conf_fail_timeout * 1000000000LL;
    lock.unlock();

    const int64_t now = this->clock();
    const int signal = exit_signal.load();

    lock.lock(this->m_pq_mutex);
    if(this->m_pq.empty() && signal)
      return;

    // When no signal is pending, wait until a fiber can be scheduled.
    if(!signal) {
      int64_t delta = INT_MAX;
      if(!this->m_pq.empty()) {
        delta = this->m_pq.front()->check_time - now;
        if(delta > 0) {
          // Rebuild the heap when there is nothing to do.
          // Note `async_time` may be overwritten by other threads at any time, so
          // we have to copy it somewhere safe.
          for(const auto& elem : this->m_pq)
            elem->check_time = elem->async_time.load();

          ::std::make_heap(this->m_pq.begin(), this->m_pq.end(), fiber_comparator);
          POSEIDON_LOG_TRACE(("Rebuilt heap for fiber scheduler: size = $1"), this->m_pq.size());
          delta = this->m_pq.front()->check_time - now;
        }
      }

      if(delta > 0) {
        // Calculate the time to wait, using binary exponential backoff.
        delta = ::rocket::min(delta, this->m_pq_wait_ns * 2 + 1, 200000000LL);
        this->m_pq_wait_ns = delta;
        lock.unlock();

        ::timespec ts;
        ts.tv_sec = 0;
        ts.tv_nsec = (long) delta;
        ::nanosleep(&ts, nullptr);
        return;
      }

      this->m_pq_wait_ns = 0;
    }

    ::std::pop_heap(this->m_pq.begin(), this->m_pq.end(), fiber_comparator);
    auto elem = this->m_pq.back();
    ROCKET_ASSERT(elem->yield_time > 0);
    ROCKET_ASSERT(elem->check_time > 0);

    if(elem->fiber->m_state.load() == async_state_finished) {
      // Delete the fiber.
      this->m_pq.pop_back();
      lock.unlock();

      POSEIDON_LOG_TRACE(("Deleting fiber `$1` (class `$2`)"), elem->fiber, typeid(*(elem->fiber)));
      elem->fiber.reset();
      do_pool_stack(elem->sched_inner->uc_stack);
      return;
    }

    // Put the fiber back.
    elem->async_time.xadd(warn_timeout);
    elem->check_time += warn_timeout;
    ::std::push_heap(this->m_pq.begin(), this->m_pq.end(), fiber_comparator);

    recursive_mutex::unique_lock sched_lock(this->m_sched_mutex);
    elem->fiber->m_scheduler = this;
    lock.unlock();

    // Process this fiber.
    POSEIDON_LOG_TRACE((
        "Processing fiber `$1` (class `$2`): state = $3"),
        elem->fiber, typeid(*(elem->fiber)), elem->fiber->m_state.load());

    auto futr = elem->futr_opt.lock();

    int64_t real_fail_timeout;
    if(elem->fail_timeout_override <= 0)
      real_fail_timeout = fail_timeout;  // use default
    else
      real_fail_timeout = elem->fail_timeout_override;

    if((elem->fail_timeout_override == 0) && (now - elem->yield_time >= warn_timeout))
      POSEIDON_LOG_WARN((
          "Fiber `$1` (class `$2`) has been suspended for `$3` ms"),
          elem->fiber, typeid(*(elem->fiber)),
          (uint64_t) (now - elem->yield_time) / 1000000ULL);

    if((elem->fail_timeout_override == 0) && (now - elem->yield_time >= fail_timeout))
      POSEIDON_LOG_ERROR((
          "Fiber `$1` (class `$2`) has been suspended for `$3` ms",
          "This circumstance looks permanent. Please check for deadlocks."),
          elem->fiber, typeid(*(elem->fiber)),
          (uint64_t) (now - elem->yield_time) / 1000000ULL);

    // 1. If the fail timeout has been exceeded, the fiber shall be resumed anyway.
    // 2. If an exit signal is pending, all fibers shall be resumed. The process
    //    will exit after all fibers complete execution.
    // 3. If the fiber is not waiting for a future, or if the future has become
    //    ready, the fiber shall be resumed.
    if((now - elem->yield_time < real_fail_timeout) && !signal && futr && futr->empty())
      return;

    if(!elem->sched_inner->uc_stack.ss_sp) {
      POSEIDON_LOG_TRACE(("Initializing fiber `$1` (class `$2`)"), elem->fiber, typeid(*(elem->fiber)));
      ::getcontext(elem->sched_inner);
      elem->sched_inner->uc_stack = do_alloc_stack(stack_vm_size);  // may throw an exception
      elem->sched_inner->uc_link = this->m_sched_outer;

      auto fiber_function = +[](int a0, int a1) noexcept
        {
          // Unpack arguments.
          Fiber_Scheduler* self;
          int args[2] = { a0, a1 };
          ::memcpy(&self, args, sizeof(self));

          do_finish_switch_fiber(self->m_sched_asan_save);
          const auto elec = self->m_sched_self_opt.lock();
          ROCKET_ASSERT(elec);

          try {
            elec->fiber->do_abstract_fiber_on_resumed();
          }
          catch(exception& stdex) {
            POSEIDON_LOG_ERROR((
                "Fiber exception: $1",
                "[fiber class `$2`]"),
                stdex, typeid(*(elec->fiber)));
          }

          ROCKET_ASSERT(elec->fiber->m_state.load() == async_state_pending);
          elec->fiber->m_state.store(async_state_running);
          POSEIDON_LOG_TRACE(("Starting fiber `$1` (class `$2`)"), elec->fiber, typeid(*(elec->fiber)));

          try {
            elec->fiber->do_abstract_fiber_on_execution();
          }
          catch(exception& stdex) {
            POSEIDON_LOG_ERROR((
                "Fiber exception: $1",
                "[fiber class `$2`]"),
                stdex, typeid(*(elec->fiber)));
          }

          POSEIDON_LOG_TRACE(("Exited from fiber `$1` (class `$2`)"), elec->fiber, typeid(*(elec->fiber)));
          ROCKET_ASSERT(elec->fiber->m_state.load() == async_state_running);
          elec->fiber->m_state.store(async_state_finished);

          try {
            elec->fiber->do_abstract_fiber_on_suspended();
          }
          catch(exception& stdex) {
            POSEIDON_LOG_ERROR((
                "Fiber exception: $1",
                "[fiber class `$2`]"),
                stdex, typeid(*(elec->fiber)));
          }

          // Return to `m_sched_outer`.
          elec->async_time.store(self->clock());
          do_start_switch_fiber(self->m_sched_asan_save, elec->sched_inner->uc_link);
        };

      int args[2];
      Fiber_Scheduler* self = this;
      ::memcpy(args, &self, sizeof(self));
      ::makecontext(elem->sched_inner, (void (*)()) fiber_function, 2, args[0], args[1]);
    }

    // Resume this fiber...
    this->m_sched_self_opt = elem;
    POSEIDON_LOG_TRACE(("Resuming fiber `$1` (class `$2`)"), elem->fiber, typeid(*(elem->fiber)));

    do_start_switch_fiber(this->m_sched_asan_save, elem->sched_inner);
    ::swapcontext(this->m_sched_outer, elem->sched_inner);
    do_finish_switch_fiber(this->m_sched_asan_save);

    // ... and return here.
    POSEIDON_LOG_TRACE(("Suspended fiber `$1` (class `$2`)"), elem->fiber, typeid(*(elem->fiber)));
    this->m_sched_self_opt.reset();

    elem->fiber->m_scheduler = nullptr;
  }

size_t
Fiber_Scheduler::
size() const noexcept
  {
    plain_mutex::unique_lock lock(this->m_pq_mutex);
    return this->m_pq.size();
  }

void
Fiber_Scheduler::
insert(unique_ptr<Abstract_Fiber>&& fiber)
  {
    if(!fiber)
      POSEIDON_THROW(("Null fiber pointer not valid"));

    const int64_t now = this->clock();

    // Create the management node.
    auto elem = ::std::make_shared<Queued_Fiber>();
    elem->fiber = ::std::move(fiber);
    elem->async_time.store(now);
    elem->yield_time = now;
    elem->check_time = now;

    // Insert it.
    plain_mutex::unique_lock lock(this->m_pq_mutex);
    this->m_pq.emplace_back(::std::move(elem));
    ::std::push_heap(this->m_pq.begin(), this->m_pq.end(), fiber_comparator);
  }

Abstract_Fiber*
Fiber_Scheduler::
self_opt() const noexcept
  {
    // Get the current fiber.
    recursive_mutex::unique_lock sched_lock(this->m_sched_mutex);

    auto elem = this->m_sched_self_opt.lock();
    if(!elem)
      return nullptr;

    ROCKET_ASSERT(elem->fiber);
    return elem->fiber.get();
  }

void
Fiber_Scheduler::
checked_yield(const Abstract_Fiber* current, const shared_ptr<Abstract_Future>& futr_opt, int64_t fail_timeout_override)
  {
    // Get the current fiber.
    recursive_mutex::unique_lock sched_lock(this->m_sched_mutex);

    auto elem = this->m_sched_self_opt.lock();
    if(!elem)
      POSEIDON_THROW(("Cannot yield execution outside a fiber"));

    if(elem->fiber.get() != current)
      POSEIDON_THROW(("Cannot yield execution outside the current fiber"));

    // If a future is given, lock it, in order to prevent race conditions.
    plain_mutex::unique_lock future_lock;
    if(futr_opt)
      future_lock.lock(futr_opt->m_init_mutex);

    // If the future is not empty, don't block at all.
    if(futr_opt && !futr_opt->empty())
      return;

    // Set the first timeout value.
    plain_mutex::unique_lock lock(this->m_conf_mutex);
    const int64_t warn_timeout = this->m_conf_warn_timeout * 1000000000LL;
    const int64_t fail_timeout = this->m_conf_fail_timeout * 1000000000LL;
    lock.unlock();

    const int64_t now = this->clock();

    int64_t real_check_timeout;
    if(fail_timeout_override <= 0)
      real_check_timeout = ::rocket::min(warn_timeout, fail_timeout);  // use default
    else
      real_check_timeout = ::rocket::min(fail_timeout_override, 0x7F000000'00000000LL - now);

    elem->async_time.store(now + real_check_timeout);
    elem->futr_opt = futr_opt;
    elem->yield_time = now;
    elem->fail_timeout_override = fail_timeout_override;

    // Attach this fiber to the wait queue of the future.
    // Other threads may update the `async_time` field to affect scheduling.
    // Typically, zero is written so the fiber can be resumed immediately.
    if(futr_opt)
      futr_opt->m_waiters.emplace_back(shared_ptr<atomic_relaxed<int64_t>>(elem, &(elem->async_time)));

    future_lock.unlock();
    elem->fiber->do_abstract_fiber_on_suspended();

    // Suspend the current fiber...
    ROCKET_ASSERT(elem->fiber->m_state.load() == async_state_running);
    elem->fiber->m_state.store(async_state_suspended);
    POSEIDON_LOG_TRACE(("Suspending fiber `$1` (class `$2`)"), elem->fiber, typeid(*(elem->fiber)));

    do_start_switch_fiber(this->m_sched_asan_save, this->m_sched_outer);
    ::swapcontext(elem->sched_inner, this->m_sched_outer);
    do_finish_switch_fiber(this->m_sched_asan_save);

    // ... and return here.
    POSEIDON_LOG_TRACE(("Resumed fiber `$1` (class `$2`)"), elem->fiber, typeid(*(elem->fiber)));
    ROCKET_ASSERT(elem->fiber->m_state.load() == async_state_suspended);
    elem->fiber->m_state.store(async_state_running);

    // Disassociate the future, if any.
    elem->futr_opt.reset();
    elem->fiber->do_abstract_fiber_on_resumed();
  }

}  // namespace poseidon
