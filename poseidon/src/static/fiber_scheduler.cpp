// This file is part of Poseidon.
// Copyleft 2020, LH_Mouse. All wrongs reserved.

#include "../precompiled.hpp"
#include "fiber_scheduler.hpp"
#include "main_config.hpp"
#include "../core/abstract_fiber.hpp"
#include "../core/config_file.hpp"
#include "../utilities.hpp"
#include <sys/resource.h>
#include <sys/mman.h>

namespace poseidon {
namespace {

int64_t
do_get_monotonic_seconds()
noexcept
  {
    ::timespec ts;
    ::clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec;
  }

// Virtual memory management for fiber stacks
const size_t page_size = static_cast<size_t>(::sysconf(_SC_PAGESIZE));

size_t
do_validate_stack_vm_size(size_t stack_vm_size)
  {
    if(stack_vm_size & 0xFFFF)
      POSEIDON_THROW("Stack size `$1` not a multiple of 64KiB", stack_vm_size);

    uintptr_t msize = (stack_vm_size >> 16) - 1;
    if(msize > 0xFFF)
      POSEIDON_THROW("Stack size `$1` out of range", stack_vm_size);

    if(stack_vm_size < page_size * 4)
      POSEIDON_THROW("Stack size `$1` less than 4 pages", stack_vm_size);

    return stack_vm_size;
  }

struct Stack_Pointer
  {
    void* base;
    size_t size;

    constexpr
    Stack_Pointer(nullptr_t = nullptr)
    noexcept
      : base(nullptr), size(0)
      { }

    constexpr
    Stack_Pointer(const ::stack_t& st)
      : base(st.ss_sp), size(st.ss_size)
      { }

    constexpr
    Stack_Pointer(void* xbase, size_t xsize)
    noexcept
      : base(xbase), size(xsize)
      { }

    explicit constexpr operator
    bool()
    const noexcept
      { return bool(this->base);  }

    operator
    ::stack_t()
    const noexcept
      {
        ::stack_t st;
        st.ss_sp = this->base;
        st.ss_size = this->size;
        return st;
      }
  };

mutex s_stack_pool_mutex;
Stack_Pointer s_stack_pool_head;

void
do_unmap_stack_aux(Stack_Pointer sp)
noexcept
  {
    char* vm_base = static_cast<char*>(sp.base) - page_size;
    size_t vm_size = sp.size + page_size * 2;

    // Note that on Linux `munmap()` may fail with `ENOMEM`.
    // There is little we can do so we ignore this error.
    if(::munmap(vm_base, vm_size) != 0)
      POSEIDON_LOG_FATAL("Could not deallocate virtual memory (base `$2`, size `$3`)\n"
                         "[`munmap()` failed: $1]",
                         noadl::format_errno(errno), vm_base, vm_size);
  }

struct Stack_Closer
  {
    constexpr
    Stack_Pointer
    null()
    const noexcept
      { return nullptr;  }

    constexpr
    bool
    is_null(Stack_Pointer sp)
    const noexcept
      { return sp.base == nullptr;  }

    void
    close(Stack_Pointer sp)
      {
        mutex::unique_lock lock(s_stack_pool_mutex);

        // Insert the region at the beginning.
        auto qnext = static_cast<Stack_Pointer*>(sp.base);
        qnext = ::rocket::construct_at(qnext, s_stack_pool_head);
        s_stack_pool_head = sp;
      }
  };

using poolable_stack = ::rocket::unique_handle<Stack_Pointer, Stack_Closer>;

poolable_stack
do_allocate_stack(size_t stack_vm_size)
  {
    Stack_Pointer sp;
    char* vm_base;
    size_t vm_size = do_validate_stack_vm_size(stack_vm_size);

    // Check whether we can get a region from the pool.
    for(;;) {
      mutex::unique_lock lock(s_stack_pool_mutex);

      sp = s_stack_pool_head;
      if(ROCKET_UNEXPECT(!sp))
        break;

      // Remove this region from the pool.
      auto qnext = static_cast<Stack_Pointer*>(sp.base);
      s_stack_pool_head = *qnext;
      ::rocket::destroy_at(qnext);

      lock.unlock();

      // Use this region if it is large enough.
      if(ROCKET_EXPECT(sp.size + page_size * 2 >= vm_size))
        return poolable_stack(sp);

      // Unmap this region and try the next one.
      do_unmap_stack_aux(sp);
    }

    // Allocate a new region with guard pages, if the pool has been exhausted.
    // Note `mmap()` returns `MAP_FAILED` upon failure, which is not a null pointer.
    vm_base = static_cast<char*>(::mmap(nullptr, vm_size, PROT_NONE,
                                        MAP_PRIVATE | MAP_ANONYMOUS | MAP_STACK, -1, 0));
    if(vm_base == MAP_FAILED)
      POSEIDON_THROW("Could not allocate virtual memory (size `$2`)\n"
                     "[`mmap()` failed: $1]",
                     noadl::format_errno(errno), vm_size);

    sp.base = vm_base + page_size;
    sp.size = vm_size - page_size * 2;
    auto sp_guard = ::rocket::make_unique_handle(sp, do_unmap_stack_aux);

    // Mark stack area writable.
    if(::mprotect(sp.base, sp.size, PROT_READ | PROT_WRITE) != 0)
      POSEIDON_THROW("Could not set stack memory permission (base `$2`, size `$3`)\n"
                     "[`mprotect()` failed: $1]",
                     noadl::format_errno(errno), sp.base, sp.size);

    // The stack need not be unmapped once all permissions have been set.
    return poolable_stack(sp_guard.release());
  }

struct Config_Scalars
  {
    size_t stack_vm_size = 0x2'00000;  // 2MiB
    int64_t warn_timeout = 15;  // 15sec
    int64_t fail_timeout = 300;  // 5min
  };

struct Thread_Context
  {
    Abstract_Fiber* current = nullptr;
    ::ucontext_t return_uctx[1];
  };

}  // namespace

POSEIDON_STATIC_CLASS_DEFINE(Fiber_Scheduler)
  {
    // constant data
    ::rocket::once_flag m_init_once;
    ::pthread_key_t m_sched_key;

    // configuration
    mutable mutex m_conf_mutex;
    Config_Scalars m_conf;

    // dynamic data
    struct Fiber_List_root
      {
        Abstract_Fiber* head = nullptr;
        Abstract_Fiber* tail = nullptr;
      };

    mutable mutex m_sched_mutex;
    condition_variable m_sched_avail;
    Fiber_List_root m_sched_ready_q;  // ready queue
    Fiber_List_root m_sched_sleep_q;  // sleep queue

    static
    void
    do_init_once()
      {
        // Create a thread-specific key for the per-thread context.
        // Note it is never destroyed.
        ::pthread_key_t ckey[1];
        int err = ::pthread_key_create(ckey, nullptr);
        if(err != 0)
          POSEIDON_THROW("Failed to allocate thread-specific key for fibers\n"
                         "[`pthread_key_create()` failed: $1]",
                         noadl::format_errno(err));

        auto key_guard = ::rocket::make_unique_handle(ckey,
                               [](::pthread_key_t* ptr) { ::pthread_key_delete(*ptr);  });

        // Set up initialized data.
        mutex::unique_lock lock(self->m_sched_mutex);
        self->m_sched_key = *(key_guard.release());
      }

    static
    Thread_Context*
    get_thread_context()
      {
        auto ptr = ::pthread_getspecific(self->m_sched_key);
        if(ROCKET_EXPECT(ptr))
          return static_cast<Thread_Context*>(ptr);

        POSEIDON_LOG_ERROR("No fiber scheduler thread context (wrong thread?)");
        return nullptr;
      }

    static
    Thread_Context*
    open_thread_context()
      {
        auto ptr = ::pthread_getspecific(self->m_sched_key);
        if(ROCKET_EXPECT(ptr))
          return static_cast<Thread_Context*>(ptr);

        // Allocate it if one hasn't been allocated yet.
        auto qctx = ::rocket::make_unique<Thread_Context>();
        POSEIDON_LOG_DEBUG("Created new fiber scheduler thread context `$1`", qctx);

        int err = ::pthread_setspecific(self->m_sched_key, qctx);
        if(err != 0)
          POSEIDON_THROW("Could not set fiber scheduler thread context\n"
                         "[`pthread_setspecific()` failed: $1]",
                         noadl::format_errno(err));

        return qctx.release();
      }

    static
    Abstract_Fiber*
    sched_dequeue_opt(Fiber_List_root& root)
    noexcept
      {
        auto fiber = root.head;
        if(!fiber)
          return nullptr;

        auto next = fiber->m_sched_next;
        (next ? next->m_sched_prev : root.tail) = nullptr;
        root.head = next;
        return fiber;
      }

    static
    void
    sched_enqueue(Fiber_List_root& root, Abstract_Fiber* fiber)
    noexcept
      {
        ROCKET_ASSERT(fiber);

        auto prev = ::std::exchange(root.tail, fiber);
        (prev ? prev->m_sched_next : root.head) = fiber;
        fiber->m_sched_next = nullptr;
        fiber->m_sched_prev = prev;
      }

    static
    void
    do_execute_fiber(int param_0, int param_1)
    noexcept
      {
        Abstract_Fiber* fiber;

        // Decode the `this` pointer to the fiber.
        int params[2] = { param_0, param_1 };
        ::std::memcpy(&fiber, params, sizeof(fiber));

        // Execute the fiber.
        ROCKET_ASSERT(fiber->state() == async_state_suspended);
        fiber->do_set_state(async_state_running);
        fiber->do_on_start();
        POSEIDON_LOG_TRACE("Starting execution of fiber `$1`", fiber);

        try {
          fiber->do_execute();
        }
        catch(exception& stdex) {
          POSEIDON_LOG_WARN("Caught an exception thrown from fiber: $1\n"
                            "[fiber class `$2`]",
                            stdex.what(), typeid(*fiber).name());
        }

        ROCKET_ASSERT(fiber->state() == async_state_running);
        fiber->do_set_state(async_state_finished);
        fiber->do_on_finish();
        POSEIDON_LOG_TRACE("Finished execution of fiber `$1`", fiber);
      }

    static
    void
    do_thread_loop(void* param)
      {
        const auto& exit_sig = *(const volatile ::std::atomic<int>*)param;
        const auto myctx = self->open_thread_context();

        rcptr<Abstract_Fiber> fiber;
        poolable_stack stack;

        // Reload configuration.
        mutex::unique_lock lock(self->m_conf_mutex);
        const auto conf = self->m_conf;
        lock.unlock();

        // Try getting a fiber from ready queue.
        lock.assign(self->m_sched_mutex);
        for(;;) {
          fiber.reset(self->sched_dequeue_opt(self->m_sched_ready_q));
          if(!fiber) {
            // Check for exit condition.
            if(exit_sig.load(::std::memory_order_relaxed) != 0)
              return;

            // Move all fibers from sleep queue to ready queue, then wait a moment.
            ::std::swap(self->m_sched_ready_q, self->m_sched_sleep_q);
            self->m_sched_avail.wait_for(lock, 100);
            continue;
          }

          auto futr = fiber->m_sched_futr;
          if(futr && (futr->state() == future_state_empty)) {
            // Check timeouts.
            int64_t now = do_get_monotonic_seconds();
            int64_t delta = now - fiber->m_sched_time;

            // Print a warning message if the fiber has been suspended for too long.
            if(now - fiber->m_sched_warn >= conf.warn_timeout) {
              fiber->m_sched_warn = now;

              POSEIDON_LOG_WARN("Fiber `$1` has been suspended for `$2` seconds.",
                                fiber, delta);
            }

            if(delta < conf.fail_timeout) {
              // Wait for the future to be set.
              self->sched_enqueue(self->m_sched_sleep_q, fiber.release());
              continue;
            }

            // Proceed anyway.
            // This usually causes an exception to be thrown after `yield()` returns.
            POSEIDON_LOG_ERROR("Suspension of fiber `$1` has exceeded `$2` seconds.\n"
                               "This is likely permanent. Please check for deadlocks.",
                               fiber, conf.fail_timeout);
          }

          ROCKET_ASSERT(fiber->state() != async_state_initial);
          if(fiber->state() == async_state_pending) {
            // This indicates whether a stack has been allocated for the fiber.
            // If the stack is in use, we cannot deallocate it safely.
            // As such, the fiber shall not be deleted until it completes execution.
            if(fiber.unique() && !fiber->resident()) {
              // Delete this fiber when no other reference of it exists.
              POSEIDON_LOG_DEBUG("Killed orphan fiber: $1", fiber);
              continue;
            }

            // Perform some initialization that might throw exceptions here.
            try {
              stack = do_allocate_stack(conf.stack_vm_size);
            }
            catch(exception& stdex) {
              POSEIDON_LOG_ERROR("Failed to initialize fiber: $1", stdex.what());

              // The fiber cannot be scheduled.
              // Put it back to the sleep queue.
              self->sched_enqueue(self->m_sched_sleep_q, fiber.release());
              continue;
            }

            // No exception shall be thrown until the end of this block.
            int ret = ::getcontext(fiber->m_sched_uctx);
            ROCKET_ASSERT(ret == 0);

            fiber->m_sched_uctx->uc_link = myctx->return_uctx;
            fiber->m_sched_uctx->uc_stack = stack.release();

            // Encode the `this` pointer to the fiber.
            int params[2] = { };
            ::std::memcpy(params, &fiber, sizeof(fiber));

            ::makecontext(fiber->m_sched_uctx,
                          reinterpret_cast<void (*)()>(do_execute_fiber),
                          2, params[0], params[1]);

            // Finish initialization.
            // Note this is the only scenerio where the fiber state is not updated
            // by itself.
            fiber->do_set_state(async_state_suspended);
          }

          // Use it.
          break;
        }
        lock.unlock();

        // Resume fhe fiber...
        ROCKET_ASSERT(fiber->state() == async_state_suspended);
        myctx->current = fiber;
        POSEIDON_LOG_TRACE("Resuming execution of fiber `$1`", fiber);

        int ret = ::swapcontext(myctx->return_uctx, fiber->m_sched_uctx);
        ROCKET_ASSERT(ret == 0);

        // ... and return here.
        myctx->current = nullptr;
        POSEIDON_LOG_TRACE("Suspended execution of fiber `$1`", fiber);

        if(fiber->state() == async_state_suspended) {
          // Put it back to the sleep queue.
          lock.assign(self->m_sched_mutex);
          self->sched_enqueue(self->m_sched_sleep_q, fiber.release());
          return;
        }

        // Otherwise, the fiber shall have completed execution.
        // Free its stack. The fiber can be safely deleted thereafter.
        ROCKET_ASSERT(fiber->state() == async_state_finished);
        stack.reset(fiber->m_sched_uctx->uc_stack);
      }
  };

void
Fiber_Scheduler::
modal_loop(const volatile ::std::atomic<int>& exit_sig)
  {
    // Perform initialization as necessary.
    self->m_init_once.call(self->do_init_once);

    // Schedule fibers and block until `exit_sig` becomes non-zero.
    while(exit_sig.load(::std::memory_order_relaxed) == 0)
      self->do_thread_loop((void*)&exit_sig);
  }

void
Fiber_Scheduler::
reload()
  {
    // Load fiber settings into temporary objects.
    auto file = Main_Config::copy();
    Config_Scalars conf;

    if(const auto qval = file.get_int64_opt({"fiber","stack_vm_size"})) {
      // Clamp the stack size between 256KiB and 256MiB for safety.
      // The upper bound (256MiB) is a hard limit, because we encode the number of
      // 64KiB chunks inside the pointer itself, so we can have at most 4096 64KiB
      // pages, which makes up 256MiB in total.
      int64_t rval = ::rocket::clamp(*qval, 0x4'0000, 0x1000'0000);
      if(*qval != rval)
        POSEIDON_LOG_WARN("Config value `fiber.stack_vm_size` truncated to `$1`\n"
                          "[value `$2` out of range]",
                          rval, *qval);

      conf.stack_vm_size = static_cast<size_t>(rval);
    }
    else {
      // Get system thread stack size.
      ::rlimit rlim;
      if(::getrlimit(RLIMIT_STACK, &rlim) != 0)
        POSEIDON_THROW("Could not get thread stack size\n"
                       "[`getrlimit()` failed: $1]",
                       noadl::format_errno(errno));

      conf.stack_vm_size = static_cast<size_t>(rlim.rlim_cur);
    }
    do_validate_stack_vm_size(conf.stack_vm_size);

    // Note a negative value indicates an infinite timeout.
    if(const auto qval = file.get_int64_opt({"fiber","warn_timeout"}))
      conf.warn_timeout = (*qval < 0) ? INT64_MAX : *qval;

    if(const auto qval = file.get_int64_opt({"fiber","fail_timeout"}))
      conf.fail_timeout = (*qval < 0) ? INT64_MAX : *qval;

    // During destruction of temporary objects the mutex should have been unlocked.
    // The swap operation is presumed to be fast, so we don't hold the mutex
    // for too long.
    mutex::unique_lock lock(self->m_conf_mutex);
    self->m_conf = conf;
  }

Abstract_Fiber*
Fiber_Scheduler::
current_opt()
noexcept
  {
    const auto myctx = self->get_thread_context();
    if(!myctx)
      return nullptr;

    auto fiber = myctx->current;
    if(!fiber)
      return nullptr;

    ROCKET_ASSERT(fiber->state() == async_state_running);
    return fiber;
  }

void
Fiber_Scheduler::
yield(rcptr<const Abstract_Future> futr_opt)
  {
    const auto myctx = self->get_thread_context();
    if(!myctx)
      POSEIDON_THROW("Invalid call to `yield()` inside a non-scheduler thread");

    auto fiber = myctx->current;
    if(!fiber)
      POSEIDON_THROW("Invalid call to `yield()` outside a fiber");

    // Suspend the current fiber...
    ROCKET_ASSERT(fiber->state() == async_state_running);
    fiber->do_set_state(async_state_suspended);
    fiber->do_on_suspend();
    POSEIDON_LOG_TRACE("Suspending execution of fiber `$1`", fiber);

    int64_t now = do_get_monotonic_seconds();
    fiber->m_sched_time = now;
    fiber->m_sched_warn = now;
    fiber->m_sched_futr = futr_opt.get();

    int ret = ::swapcontext(fiber->m_sched_uctx, myctx->return_uctx);
    ROCKET_ASSERT(ret == 0);

    // ... and resume here.
    ROCKET_ASSERT(myctx->current == fiber);
    ROCKET_ASSERT(fiber->state() == async_state_suspended);
    fiber->do_set_state(async_state_running);
    fiber->do_on_resume();
    POSEIDON_LOG_TRACE("Resumed execution of fiber `$1`", fiber);
  }

rcptr<Abstract_Fiber>
Fiber_Scheduler::
insert(uptr<Abstract_Fiber>&& ufiber)
  {
    // Take ownership of `ufiber`.
    rcptr<Abstract_Fiber> fiber(ufiber.release());
    if(!fiber)
      POSEIDON_THROW("Null fiber pointer not valid");

    if(!fiber.unique())
      POSEIDON_THROW("Fiber pointer must be unique");

    // Clear data members.
    fiber->m_sched_time = 0;
    fiber->m_sched_warn = 0;
    fiber->m_sched_futr = nullptr;

    // Lock fiber queue for modification.
    mutex::unique_lock lock(self->m_sched_mutex);

    // Insert this fiber at the end of ready queue.
    self->sched_enqueue(self->m_sched_ready_q, fiber);
    fiber->add_reference();
    fiber->do_set_state(async_state_pending);
    self->m_sched_avail.notify_one();
    return fiber;
  }

void
Fiber_Scheduler::
signal()
noexcept
  {
    mutex::unique_lock lock(self->m_sched_mutex);
    self->m_sched_avail.notify_one();
  }

}  // namespace poseidon
