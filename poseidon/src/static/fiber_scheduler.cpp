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

struct VM_ptr
  {
    uintptr_t bits;

    constexpr
    VM_ptr(nullptr_t = nullptr)
    noexcept
      : bits(0)
      { }

    VM_ptr(void* base, size_t size)
    noexcept
      {
        // Validate that `base` is aligned to a 4KiB boundary.
        uintptr_t mbase = reinterpret_cast<uintptr_t>(base);
        ROCKET_ASSERT_MSG((mbase & 0xFFF) == 0, "unaligned mapping base address");

        // Validate that `size` is a multiple of 64KiB and is not zero.
        uintptr_t msize = (size >> 16) - 1;
        ROCKET_ASSERT_MSG(((msize & 0xFFF) + 1) << 16 == size, "invalid mapping size");

        this->bits = mbase | msize;
      }

    explicit constexpr operator
    bool()
    const noexcept
      { return this->bits != 0;  }

    void*
    base()
    const noexcept
      { return reinterpret_cast<void*>(this->bits & uintptr_t(~0xFFF));  }

    size_t
    size()
    const noexcept
      { return ((this->bits & 0xFFF) + 1) << 16;  }
  };

const size_t page_size = static_cast<size_t>(::sysconf(_SC_PAGESIZE));
mutex s_stack_pool_mutex;
VM_ptr s_stack_pool_head;

size_t
do_validate_stack_size(size_t stack_size)
  {
    if(stack_size & 0xFFFF)
      POSEIDON_THROW("stack size `$1` not a multiple of 64KiB", stack_size);

    uintptr_t msize = (stack_size >> 16) - 1;
    if(msize > 0xFFF)
      POSEIDON_THROW("stack size `$1` out of range", stack_size);

    if(stack_size < page_size * 4)
      POSEIDON_THROW("stack size `$1` less than 4 pages", stack_size);

    return stack_size;
  }

void
do_alloc_stack(::stack_t& stack, size_t stack_size)
  {
    const auto vmp_unmap = [](VM_ptr* mp) { ::munmap(mp->base(), mp->size());  };
    size_t size = do_validate_stack_size(stack_size);
    VM_ptr vmp;

    for(;;) {
      // Check whether we can get a block from the pool.
      mutex::unique_lock lock(s_stack_pool_mutex);

      vmp = s_stack_pool_head;
      if(ROCKET_UNEXPECT(!vmp))
        break;

      // Exclude guard pages from both ends.
      ROCKET_ASSERT(vmp.size() >= page_size * 2);
      stack.ss_sp = static_cast<char*>(vmp.base()) + page_size;
      stack.ss_size = vmp.size() - page_size * 2;

      // Remove it from the pool.
      auto qnext = reinterpret_cast<VM_ptr*>(stack.ss_sp);
      s_stack_pool_head = ::std::move(*qnext);
      ::rocket::destroy_at(qnext);

      lock.unlock();

      // Return this block if it is large enough.
      if(ROCKET_EXPECT(vmp.size() >= stack_size))
        return;

      // Unmap this block and try the next one.
      vmp_unmap(&vmp);
    }

    // Map a new block (including guard pages) if the pool has been exhausted.
    // Note `mmap()` returns `MAP_FAILED` upon failure, which is not a null pointer.
    void* base = ::mmap(nullptr, size, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_STACK,
                                       -1, 0);
    if(base == MAP_FAILED) {
      POSEIDON_THROW("error allocating virtual memory (size `$2`)\n"
                     "[`mmap()` failed: $1]",
                     noadl::format_errno(errno), size);
    }
    vmp = { base, size };
    uptr<VM_ptr, decltype((vmp_unmap))> vmp_guard(&vmp, vmp_unmap);

    // Exclude guard pages from both ends.
    ROCKET_ASSERT(vmp.size() >= page_size * 2);
    stack.ss_sp = static_cast<char*>(vmp.base()) + page_size;
    stack.ss_size = vmp.size() - page_size * 2;

    // Mark stack area writable.
    if(::mprotect(stack.ss_sp, stack.ss_size, PROT_READ | PROT_WRITE) != 0)
      POSEIDON_THROW("error changing stack memory permission (base `$2`, size `$3`)\n"
                     "[`mprotect()` failed: $1]",
                     noadl::format_errno(errno), stack.ss_sp, stack.ss_size);

    // Release its ownership.
    // The block is never unmapped. It will be pooled for reuse by later fibers.
    vmp_guard.release();
  }

void
do_free_stack(const ::stack_t& stack)
noexcept
  {
    if(!stack.ss_sp)
      return;

    // Put the block back into the pool.
    mutex::unique_lock lock(s_stack_pool_mutex);

    void* base = static_cast<char*>(stack.ss_sp) - page_size;
    size_t size = stack.ss_size + page_size * 2;

    auto qnext = reinterpret_cast<VM_ptr*>(stack.ss_sp);
    qnext = ::rocket::construct_at(qnext, ::std::move(s_stack_pool_head));
    s_stack_pool_head = { base, size };
  }

struct Config_Scalars
  {
    size_t stack_size = 0x2'00000;  // 2MiB
    int64_t warn_timeout = 600; // 10min
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
    Fiber_List_root m_sched_ready_q;  // ready queue
    Fiber_List_root m_sched_sleep_q;  // sleep queue

    static
    void
    do_init_once()
      {
        // Create a thread-specific key for the per-thread context.
        const auto delete_context = [](void* ptr) { delete static_cast<Thread_Context*>(ptr);  };
        const auto delete_key = [](::pthread_key_t* ptr) { ::pthread_key_delete(*ptr);  };

        ::pthread_key_t ckey[1];
        int err = ::pthread_key_create(ckey, delete_context);
        if(err != 0)
          POSEIDON_THROW("error allocating thread-specific key for fiber scheduling\n"
                         "[`pthread_key_create()` failed: $1]",
                         noadl::format_errno(err));
        uptr<::pthread_key_t, decltype((delete_key))> key_guard(ckey, delete_key);

        // Set up initialized data.
        mutex::unique_lock lock(self->m_sched_mutex);
        self->m_sched_key = *(key_guard.release());
      }

    static
    Thread_Context*
    open_thread_context()
      {
        auto ptr = ::pthread_getspecific(self->m_sched_key);
        if(ROCKET_EXPECT(ptr))
          return static_cast<Thread_Context*>(ptr);

        // Allocate it if one hasn't been allocated yet.
        auto uctx = ::rocket::make_unique<Thread_Context>();
        POSEIDON_LOG_DEBUG("Created new fiber scheduler thread context `$1`", uctx);

        int err = ::pthread_setspecific(self->m_sched_key, uctx.get());
        if(err != 0)
          POSEIDON_THROW("could not set fiber scheduler thread context\n"
                         "[`pthread_setspecific()` failed: $1]",
                         noadl::format_errno(err));
        return uctx.release();
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
    do_fiber_execute(int param_0, int param_1)
      {
        Abstract_Fiber* fiber;

        // Decode the `this` pointer.
        int params[2] = { param_0, param_1 };
        ::std::memcpy(&fiber, params, sizeof(fiber));

        // Execute the fiber.
        POSEIDON_LOG_TRACE("Fiber `$1` started", fiber);
        try {
          fiber->do_execute();
          POSEIDON_LOG_TRACE("Fiber `$1` finished", fiber);
        }
        catch(exception& stdex) {
          // Ignore this exeption.
          POSEIDON_LOG_WARN("Fiber `$1` threw an exception: $2", fiber, stdex.what());
        }
        ROCKET_ASSERT(fiber->m_state.load(::std::memory_order_relaxed) == async_state_running);
        fiber->m_state.store(async_state_finished, ::std::memory_order_relaxed);
      }

    static
    void
    do_thread_loop(void* param)
      {
        const auto& exit_sig = *(const volatile ::std::atomic<int>*)param;
        const auto myctx = self->open_thread_context();
        Abstract_Fiber* fiber;

        // Reload configuration.
        mutex::unique_lock lock(self->m_conf_mutex);
        const auto conf = self->m_conf;
        lock.unlock();

        // Try getting a fiber from ready queue.
        lock.assign(self->m_sched_mutex);
        fiber = self->sched_dequeue_opt(self->m_sched_ready_q);
        if(!fiber) {
          ::std::swap(self->m_sched_ready_q, self->m_sched_sleep_q);
          lock.unlock();

          // Wait a moment.
          if(exit_sig.load(::std::memory_order_relaxed) == 0)
            ::usleep(200'000);
          return;
        }
        lock.unlock();

        // Perform some initialization if the fiber is in pending state.
        if(fiber->state() == async_state_pending) {
          fiber->m_sched_time = 0;
          fiber->m_sched_warn = 0;
          fiber->m_sched_futr = nullptr;

          int ret = ::getcontext(fiber->m_sched_uctx);
          ROCKET_ASSERT(ret == 0);

          try {
            do_alloc_stack(fiber->m_sched_uctx->uc_stack, conf.stack_size);
          }
          catch(exception& stdex) {
            POSEIDON_LOG_ERROR("Failed to allocate fiber stack of size `$1`: $2",
                               conf.stack_size, stdex.what());

            // Put the fiber back into sleep queue.
            self->sched_enqueue(self->m_sched_sleep_q, fiber);
            return;
          }
          fiber->m_sched_uctx->uc_link = myctx->return_uctx;

          // Encode the `this` pointer.
          int params[2];
          ::std::memcpy(params, &fiber, sizeof(fiber));

          ::makecontext(fiber->m_sched_uctx, reinterpret_cast<void (*)()>(do_fiber_execute),
                                             2, params[0], params[1]);

          fiber->m_state.store(async_state_suspended, ::std::memory_order_relaxed);
        }

        // Check for blocking conditions.
        if(fiber->m_sched_futr && (fiber->m_sched_futr->state() == future_state_empty)) {
          // Print a warning message if the fiber has been suspended for too long.
          int64_t now = do_get_monotonic_seconds();
          if(now - fiber->m_sched_warn >= conf.warn_timeout) {
            fiber->m_sched_warn = now;

            POSEIDON_LOG_WARN(
                "Fiber `$1` has been suspended for `$2` seconds, which seems too long",
                fiber, now - fiber->m_sched_time);
          }

          // Put the fiber back into sleep queue.
          self->sched_enqueue(self->m_sched_sleep_q, fiber);
          return;
        }

        ROCKET_ASSERT(fiber->state() == async_state_suspended);
        fiber->m_state.store(async_state_running, ::std::memory_order_relaxed);

        // Resume the fiber...
        myctx->current = fiber;

        POSEIDON_LOG_TRACE("Control flow entering fiber `$1`", fiber);
        int ret = ::swapcontext(myctx->return_uctx, fiber->m_sched_uctx);
        ROCKET_ASSERT(ret == 0);
        POSEIDON_LOG_TRACE("Control flow left fiber `$1`", fiber);

        myctx->current = nullptr;

        // ... and check whether it returned or was suspended.
        if(ROCKET_EXPECT(fiber->state() == async_state_finished)) {
          // Delete the fiber from scheduler.
          do_free_stack(fiber->m_sched_uctx->uc_stack);
          static_cast<void>(rcptr<Abstract_Fiber>(fiber));
          return;
        }

        ROCKET_ASSERT(fiber->state() == async_state_running);

        // Put the fiber back into sleep queue.
        fiber->m_state.store(async_state_suspended, ::std::memory_order_relaxed);
        lock.assign(self->m_sched_mutex);
        self->sched_enqueue(self->m_sched_sleep_q, fiber);
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

    if(const auto qval = file.get_int64_opt({"fiber","stack_size"})) {
      // Clamp the stack size between 256KiB and 256MiB for safety.
      // The upper bound (256MiB) is a hard limit, because we encode the number of
      // 64KiB chunks inside the pointer itself, so we can have at most 4096 64KiB
      // pages, which makes up 256MiB in total.
      int64_t rval = ::rocket::clamp(*qval, 0x4'0000, 0x1000'0000);
      if(*qval != rval)
        POSEIDON_LOG_WARN("Config value `fiber.stack_size` truncated to `$1`\n"
                          "[value `$2` out of range]",
                          rval, *qval);
      conf.stack_size = static_cast<size_t>(rval);
    }
    else {
      // Get system thread stack size.
      ::rlimit rlim;
      if(::getrlimit(RLIMIT_STACK, &rlim) != 0)
        POSEIDON_THROW("could not get thread stack size\n"
                       "[`getrlimit()` failed: $1]",
                       noadl::format_errno(errno));
      conf.stack_size = static_cast<size_t>(rlim.rlim_cur);
    }
    do_validate_stack_size(conf.stack_size);

    // Note a negative value indicates an infinite timeout.
    if(const auto qval = file.get_int64_opt({"fiber","warn_timeout"}))
      conf.warn_timeout = (*qval < 0) ? INT64_MAX : *qval;

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
    const auto myctx = static_cast<Thread_Context*>(::pthread_getspecific(self->m_sched_key));
    if(!myctx)
      return nullptr;

    auto fiber = myctx->current;
    if(!fiber)
      return nullptr;

    ROCKET_ASSERT(fiber->m_state.load(::std::memory_order_relaxed) == async_state_running);
    return fiber;
  }

void
Fiber_Scheduler::
yield(rcptr<const Abstract_Future> futr_opt)
  {
    const auto myctx = static_cast<Thread_Context*>(::pthread_getspecific(self->m_sched_key));
    if(!myctx)
      POSEIDON_THROW("invalid call to `yield()` inside a non-scheduler thread");

    auto fiber = myctx->current;
    if(!fiber)
      POSEIDON_THROW("invalid call to `yield()` outside a fiber");

    // Suspend the current fiber...
    ROCKET_ASSERT(fiber->m_state.load(::std::memory_order_relaxed) == async_state_running);
    int64_t now = do_get_monotonic_seconds();
    fiber->m_sched_time = now;
    fiber->m_sched_warn = now;
    fiber->m_sched_futr = futr_opt.get();

    POSEIDON_LOG_TRACE("Control flow leaving fiber `$1`", fiber);
    int ret = ::swapcontext(fiber->m_sched_uctx, myctx->return_uctx);
    ROCKET_ASSERT(ret == 0);
    POSEIDON_LOG_TRACE("Control flow entered fiber `$1`", fiber);

    // ... and resume here.
    ROCKET_ASSERT(myctx->current == fiber);
    ROCKET_ASSERT(fiber->m_state.load(::std::memory_order_relaxed) == async_state_running);
  }

rcptr<Abstract_Fiber>
Fiber_Scheduler::
insert(uptr<Abstract_Fiber>&& ufiber)
  {
    // Take ownership of `ufiber`.
    rcptr<Abstract_Fiber> fiber(ufiber.release());
    if(!fiber)
      POSEIDON_THROW("null fiber pointer not valid");

    if(!fiber.unique())
      POSEIDON_THROW("fiber pointer must be unique");

    // Lock fiber queue for modification.
    mutex::unique_lock lock(self->m_sched_mutex);

    // Insert this fiber at the end of ready queue.
    self->sched_enqueue(self->m_sched_ready_q, fiber);
    fiber->add_reference();
    fiber->m_state.store(async_state_pending, ::std::memory_order_relaxed);
    return fiber;
  }

}  // namespace poseidon
