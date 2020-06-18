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

struct PQ_Element
  {
    int64_t time;
    uint32_t version;
    rcptr<Abstract_Fiber> fiber;
  };

struct PQ_Compare
  {
    constexpr
    bool
    operator()(const PQ_Element& lhs, const PQ_Element& rhs)
    const noexcept
      { return lhs.time > rhs.time;  }

    constexpr
    bool
    operator()(const PQ_Element& lhs, int64_t rhs)
    const noexcept
      { return lhs.time > rhs;  }

    constexpr
    bool
    operator()(int64_t lhs, const PQ_Element& rhs)
    const noexcept
      { return lhs > rhs.time;  }
  }
  constexpr pq_compare;

struct Thread_Context
  {
    Abstract_Fiber* current = nullptr;
    ::ucontext_t return_uctx[1];
  };

union Fancy_Fiber_Pointer
  {
    Abstract_Fiber* fiber;
    int words[2];

    constexpr
    Fancy_Fiber_Pointer(Abstract_Fiber* xfiber)
    noexcept
      : fiber(xfiber)
      { }

    constexpr
    Fancy_Fiber_Pointer(int word_0, int word_1)
    noexcept
      : words{ word_0, word_1 }
      { }

    constexpr
    Abstract_Fiber&
    operator*()
    const noexcept
      { return *(this->fiber);  }

    constexpr
    Abstract_Fiber*
    operator->()
    const noexcept
      { return this->fiber;  }

    constexpr operator
    Abstract_Fiber*()
    const noexcept
      { return this->fiber;  }
  };

inline
tinyfmt&
operator<<(tinyfmt& fmt, const Fancy_Fiber_Pointer& fcptr)
  { return fmt << static_cast<Abstract_Fiber*>(fcptr);  }

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
    mutable mutex m_sched_mutex;
    condition_variable m_sched_avail;
    ::std::vector<PQ_Element> m_sched_pq;
    Abstract_Fiber* m_sched_recq = nullptr;

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
    void
    do_push_fiber_unchecked(int64_t now, uint32_t version, rcptr<Abstract_Fiber>&& fiber)
    noexcept
      {
        // The caller must have reserved space in the scheduler queue.
        PQ_Element elem;
        elem.time = now;
        elem.version = version;
        elem.fiber = ::std::move(fiber);

        ROCKET_ASSERT(self->m_sched_pq.size() < self->m_sched_pq.capacity());
        self->m_sched_pq.emplace_back(::std::move(elem));
        ::std::push_heap(self->m_sched_pq.begin(), self->m_sched_pq.end(), pq_compare);
      }

    static
    void
    do_execute_fiber(int word_0, int word_1)
    noexcept
      {
        Fancy_Fiber_Pointer fcptr(word_0, word_1);

        // Execute the fiber.
        ROCKET_ASSERT(fcptr->state() == async_state_suspended);
        fcptr->do_set_state(async_state_running);
        fcptr->do_on_start();
        POSEIDON_LOG_TRACE("Starting execution of fiber `$1`", fcptr);

        try {
          fcptr->do_execute();
        }
        catch(exception& stdex) {
          POSEIDON_LOG_WARN("Caught an exception thrown from fiber: $1\n"
                            "[fiber class `$2`]",
                            stdex.what(), typeid(*fcptr).name());
        }

        ROCKET_ASSERT(fcptr->state() == async_state_running);
        fcptr->do_set_state(async_state_finished);
        fcptr->do_on_finish();
        POSEIDON_LOG_TRACE("Finished execution of fiber `$1`", fcptr);
      }

    static
    void
    do_thread_loop(void* param)
      {
        const auto& exit_sig = *(const volatile ::std::atomic<int>*)param;
        const auto myctx = self->open_thread_context();

        rcptr<Abstract_Fiber> fiber;
        int64_t now;
        poolable_stack stack;

        // Reload configuration.
        mutex::unique_lock lock(self->m_conf_mutex);
        const auto conf = self->m_conf;
        lock.unlock();

        // Await a fiber and pop it.
        lock.assign(self->m_sched_mutex);
        for(;;) {
          fiber.reset();
          int sig = exit_sig.load(::std::memory_order_relaxed);
          now = do_get_monotonic_seconds();

          // Move all fibers from the recycle queue to the scheduler queue.
          self->m_sched_pq.reserve(self->m_sched_pq.size() + 100);
          while(fiber.reset(self->m_sched_recq)) {
            // Note pushing cannot throw exceptions.
            self->m_sched_recq = ::std::exchange(fiber->m_sched_next, nullptr);
            self->do_push_fiber_unchecked(now, ++(fiber->m_sched_version), ::std::move(fiber));

            // Ensure we always have some space reserved at the end after this loop.
            self->m_sched_pq.reserve(self->m_sched_pq.size() + 20);
          }

          if(sig == 0) {
            // Try popping a fiber from the scheduler queue.
            if(self->m_sched_pq.empty()) {
              // Wait until a fiber becomes available.
              self->m_sched_avail.wait_for(lock, 200);
              continue;
            }

            // Check the first element.
            int64_t delta = self->m_sched_pq.front().time - now;
            if(delta > 0) {
              self->m_sched_avail.wait_for(lock, 200);
              continue;
            }
          }
          else {
            // If a signal has been received, force execution of all fibers.
            if(self->m_sched_pq.empty()) {
              // Exit if there are no more fibers.
              POSEIDON_LOG_INFO("Shutting down due to signal $1: $2", sig, ::sys_siglist[sig]);
              ::sleep(1);
              ::std::quick_exit(0);
            }
          }

          // Pop the first fiber.
          ::std::pop_heap(self->m_sched_pq.begin(), self->m_sched_pq.end(), pq_compare);
          auto& elem = self->m_sched_pq.back();
          fiber = ::std::move(elem.fiber);

          if(elem.version != fiber->m_sched_version) {
            // Delete this invalidated element.
            self->m_sched_pq.pop_back();
            continue;
          }

          if(fiber.unique() && !fiber->resident() && (fiber->state() == async_state_pending)) {
            // Delete this fiber when no other reference of it exists.
            POSEIDON_LOG_DEBUG("Killed orphan fiber: $1", fiber);
            self->m_sched_pq.pop_back();
            continue;
          }

          // Check for blocking conditions.
          if((sig == 0) && fiber->m_sched_futp && fiber->m_sched_futp->empty()) {
            int64_t delta = now - fiber->m_sched_yield_time;
            if(delta < conf.fail_timeout) {
              // Print a warning message if the fiber has been suspended for too long.
              if(delta >= conf.warn_timeout)
                POSEIDON_LOG_WARN("Fiber `$1` has been suspended for `$2` seconds.", fiber, delta);

              // Put the fiber back into the queue.
              elem.fiber = ::std::move(fiber);
              elem.time = now + ::rocket::min(conf.warn_timeout, delta);
              ::std::push_heap(self->m_sched_pq.begin(), self->m_sched_pq.end(), pq_compare);
              continue;
            }

            // Proceed anyway.
            // This usually causes an exception to be thrown after `yield()` returns.
            POSEIDON_LOG_ERROR("Suspension of fiber `$1` has exceeded `$2` seconds.\n"
                               "This circumstance might be permanent. Please check for deadlocks.",
                               fiber, conf.fail_timeout);
          }

          // Process this fiber!
          self->m_sched_pq.pop_back();
          break;
        }
        lock.unlock();

        // Initialize the fiber stack as necessary.
        if(fiber->state() == async_state_pending) {
          // Perform some initialization that might throw exceptions here.
          try {
            stack = do_allocate_stack(conf.stack_vm_size);
          }
          catch(exception& stdex) {
            POSEIDON_LOG_ERROR("Failed to initialize fiber: $1", stdex.what());
            lock.assign(self->m_sched_mutex);
            self->do_push_fiber_unchecked(now + 1, fiber->m_sched_version, ::std::move(fiber));
            return;
          }

          int ret = ::getcontext(fiber->m_sched_uctx);
          ROCKET_ASSERT(ret == 0);

          fiber->m_sched_uctx->uc_link = myctx->return_uctx;
          fiber->m_sched_uctx->uc_stack = stack.release();

          // Initialize the fiber context.
          Fancy_Fiber_Pointer fcptr(fiber.get());

          ::makecontext(fiber->m_sched_uctx, reinterpret_cast<void (*)()>(do_execute_fiber),
                                             2, fcptr.words[0], fcptr.words[1]);

          // Finish initialization.
          // Note this is the only scenerio where the fiber state is not updated
          // by itself.
          fiber->do_set_state(async_state_suspended);
        }

        // Resume this fiber...
        ROCKET_ASSERT(fiber->state() == async_state_suspended);
        myctx->current = fiber;
        POSEIDON_LOG_TRACE("Resuming execution of fiber `$1`", fiber);

        int ret = ::swapcontext(myctx->return_uctx, fiber->m_sched_uctx);
        ROCKET_ASSERT(ret == 0);

        // ... and return here.
        myctx->current = nullptr;
        POSEIDON_LOG_TRACE("Suspended execution of fiber `$1`", fiber);

        if(fiber->state() == async_state_suspended) {
          // Push a sentinel.
          lock.assign(self->m_sched_mutex);
          self->do_push_fiber_unchecked(now + conf.warn_timeout, fiber->m_sched_version,
                                        ::std::move(fiber));
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
    for(;;)
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
      conf.warn_timeout = (*qval | *qval >> 63) & INT64_MAX;

    if(const auto qval = file.get_int64_opt({"fiber","fail_timeout"}))
      conf.fail_timeout = (*qval | *qval >> 63) & INT64_MAX;

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
yield(rcptr<const Abstract_Future> futp_opt)
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

    mutex::unique_lock lock(self->m_sched_mutex);
    fiber->m_sched_yield_time = now;
    fiber->m_sched_futp = futp_opt.get();
    if(futp_opt) {
      // Attach the fiber to the future's wait queue if one is provided.
      // The queue shall own a reference to the fiber.
      const auto& futr = *futp_opt;
      fiber->m_sched_next = ::std::exchange(futr.m_sched_head, fiber);
      fiber->add_reference();
      lock.unlock();

      int ret = ::swapcontext(fiber->m_sched_uctx, myctx->return_uctx);
      ROCKET_ASSERT(ret == 0);

      // If the fiber resumes execution because suspension timed out, remove it
      // from the future's wait queue.
      lock.assign(self->m_sched_mutex);
      for(auto qrefl = &(futr.m_sched_head);  *qrefl;
                                 qrefl = &((*qrefl)->m_sched_next))
        if(*qrefl == fiber) {
          *qrefl = fiber->m_sched_next;
          fiber->drop_reference();
          break;
        }
      lock.unlock();
    }
    else {
      // Attach the fiber to the recycle queue of the current thread otherwise.
      // The queue shall own a reference to the fiber.
      fiber->m_sched_next = ::std::exchange(self->m_sched_recq, fiber);
      fiber->add_reference();
      lock.unlock();

      int ret = ::swapcontext(fiber->m_sched_uctx, myctx->return_uctx);
      ROCKET_ASSERT(ret == 0);
    }

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

    // Attach this fiber to the recycle queue.
    mutex::unique_lock lock(self->m_sched_mutex);
    fiber->m_sched_yield_time = 0;
    fiber->m_sched_futp = nullptr;
    fiber->m_sched_next = ::std::exchange(self->m_sched_recq, fiber);
    fiber->add_reference();
    fiber->do_set_state(async_state_pending);
    self->m_sched_avail.notify_one();
    return fiber;
  }

bool
Fiber_Scheduler::
signal(const Abstract_Future& futr)
noexcept
  {
    // Lock the future's wait queue and the global recycle queue.
    mutex::unique_lock lock(self->m_sched_mutex);

    // Move all fibers from the future's wait queue to the recycle queue.
    auto qhead = futr.m_sched_head;
    if(!qhead)
      return false;

    // Locate the last node.
    auto qtail = ::std::exchange(futr.m_sched_head, nullptr);
    while(qtail->m_sched_next)
      qtail = qtail->m_sched_next;

    // Splice the two queues.
    // Fibers are moved from one queue to the other, so there is no need to
    // tamper with reference counts here.
    qtail->m_sched_next = ::std::exchange(self->m_sched_recq, qhead);
    self->m_sched_avail.notify_one();
    return true;
  }

}  // namespace poseidon
