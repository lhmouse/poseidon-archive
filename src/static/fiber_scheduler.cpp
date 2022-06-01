// This file is part of Poseidon.
// Copyleft 2020, LH_Mouse. All wrongs reserved.

#include "../precompiled.ipp"
#include "fiber_scheduler.hpp"
#include "main_config.hpp"
#include "../core/abstract_fiber.hpp"
#include "../core/config_file.hpp"
#include "../utils.hpp"
#include <sys/resource.h>
#include <sys/mman.h>
#include <semaphore.h>
#include <signal.h>

namespace poseidon {
namespace {

struct Stack_pointer
  {
    void* base;
    size_t size;

    constexpr
    Stack_pointer(nullptr_t = nullptr) noexcept
      : base(nullptr), size(0)
      { }

    constexpr
    Stack_pointer(const ::stack_t& st)
      : base(st.ss_sp), size(st.ss_size)
      { }

    constexpr
    Stack_pointer(void* xbase, size_t xsize) noexcept
      : base(xbase), size(xsize)
      { }

    explicit constexpr operator
    bool() const noexcept
      { return bool(this->base);  }

    operator
    ::stack_t() const noexcept
      {
        ::stack_t st;
        st.ss_sp = this->base;
        st.ss_size = this->size;
        return st;
      }
  };

const size_t s_page_size = static_cast<size_t>(::sysconf(_SC_PAGESIZE));
simple_mutex s_stack_pool_mutex;
Stack_pointer s_stack_pool_head;

size_t
do_validate_stack_vm_size(size_t stack_vm_size)
  {
    if(stack_vm_size & 0xFFFF)
      POSEIDON_THROW("stack size `$1` not a multiple of 64KiB", stack_vm_size);

    uintptr_t msize = (stack_vm_size >> 16) - 1;
    if(msize > 0xFFF)
      POSEIDON_THROW("stack size `$1` out of range", stack_vm_size);

    if(stack_vm_size < s_page_size * 4)
      POSEIDON_THROW("stack size `$1` less than 4 pages", stack_vm_size);

    return stack_vm_size;
  }

void
do_unmap_stack_aux(Stack_pointer sp) noexcept
  {
    char* vm_base = static_cast<char*>(sp.base) - s_page_size;
    size_t vm_size = sp.size + s_page_size * 2;

    // Note that on Linux `munmap()` may fail with `ENOMEM`.
    // There is little we can do so we ignore this error.
    if(::munmap(vm_base, vm_size) != 0)
      POSEIDON_LOG_FATAL(
          "Could not deallocate virtual memory (base `$2`, size `$3`)\n"
          "[`munmap()` failed: $1]",
          format_errno(), vm_base, vm_size);
  }

struct Stack_delete
  {
    constexpr
    Stack_pointer
    null() const noexcept
      { return nullptr;  }

    constexpr
    bool
    is_null(Stack_pointer sp) const noexcept
      { return sp.base == nullptr;  }

    void
    close(Stack_pointer sp) noexcept
      {
        simple_mutex::unique_lock lock(s_stack_pool_mutex);

        // Insert the region at the beginning.
        auto qnext = static_cast<Stack_pointer*>(sp.base);
        qnext = ::rocket::construct(qnext, s_stack_pool_head);
        s_stack_pool_head = sp;
      }
  };

using unique_stack = ::rocket::unique_handle<Stack_pointer, Stack_delete>;

unique_stack
do_allocate_stack(size_t stack_vm_size)
  {
    Stack_pointer sp;
    char* vm_base;
    size_t vm_size = do_validate_stack_vm_size(stack_vm_size);

    // Check whether we can get a region from the pool.
    for(;;) {
      simple_mutex::unique_lock lock(s_stack_pool_mutex);
      sp = s_stack_pool_head;
      if(ROCKET_UNEXPECT(!sp))
        break;

      // Remove this region from the pool.
      auto qnext = static_cast<Stack_pointer*>(sp.base);
      s_stack_pool_head = *qnext;
      ::rocket::destroy(qnext);
      lock.unlock();

      // Use this region if it is large enough.
      if(ROCKET_EXPECT(sp.size + s_page_size * 2 >= vm_size))
        return unique_stack(sp);

      // Unmap this region and try the next one.
      do_unmap_stack_aux(sp);
    }

    // Allocate a new region with guard pages, if the pool has been exhausted.
    // Note `mmap()` returns `MAP_FAILED` upon failure, which is not a null pointer.
    vm_base = static_cast<char*>(::mmap(nullptr, vm_size, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_STACK, -1, 0));
    if(vm_base == MAP_FAILED)
      POSEIDON_THROW(
          "could not allocate virtual memory (size `$2`)\n"
          "[`mmap()` failed: $1]",
          format_errno(), vm_size);

    sp.base = vm_base + s_page_size;
    sp.size = vm_size - s_page_size * 2;
    auto sp_guard = ::rocket::make_unique_handle(sp, do_unmap_stack_aux);

    // Mark stack area writable.
    if(::mprotect(sp.base, sp.size, PROT_READ | PROT_WRITE) != 0)
      POSEIDON_THROW(
          "could not set stack memory permission (base `$2`, size `$3`)\n"
          "[`mprotect()` failed: $1]",
          format_errno(), sp.base, sp.size);

    // The stack need not be unmapped once all permissions have been set.
    return unique_stack(sp_guard.release());
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
    uint32_t serial;
    rcptr<Abstract_Fiber> fiber;
    int64_t reserved;
  };

struct PQ_Compare
  {
    constexpr
    bool
    operator()(const PQ_Element& lhs, const PQ_Element& rhs) const noexcept
      { return lhs.time > rhs.time;  }

    constexpr
    bool
    operator()(const PQ_Element& lhs, int64_t rhs) const noexcept
      { return lhs.time > rhs;  }

    constexpr
    bool
    operator()(int64_t lhs, const PQ_Element& rhs) const noexcept
      { return lhs > rhs.time;  }
  }
  constexpr pq_compare;

struct Thread_Context
  {
    rcptr<Abstract_Fiber> current;
    void* asan_fiber_save;  // used by address sanitizer
    ::ucontext_t return_uctx[1];
  };

inline
void
do_start_switch_fiber(Thread_Context* myctx, ::ucontext_t* uctx) noexcept
  {
#ifdef POSEIDON_ENABLE_ADDRESS_SANITIZER
    extern "C" void __sanitizer_start_switch_fiber(void**, const void*, size_t) noexcept;
    __sanitizer_start_switch_fiber(&(myctx->asan_fiber_save), uctx->uc_stack.ss_sp, uctx->uc_stack.ss_size);
#else
    (void) myctx;
    (void) uctx;
#endif
  }

inline
void
do_finish_switch_fiber(Thread_Context* myctx) noexcept
  {
#ifdef POSEIDON_ENABLE_ADDRESS_SANITIZER
    extern "C" void __sanitizer_finish_switch_fiber(void*, const void**, size_t*) noexcept;
    __sanitizer_finish_switch_fiber(myctx->asan_fiber_save, nullptr, nullptr);
#else
    (void) myctx;
#endif
  }

union Fiber_pointer
  {
    Abstract_Fiber* fiber;
    int words[2];

    explicit constexpr
    Fiber_pointer(Abstract_Fiber* xfiber) noexcept
      : fiber(xfiber)
      { }

    explicit constexpr
    Fiber_pointer(int word_0, int word_1) noexcept
      : words{ word_0, word_1 }
      { }

    constexpr operator
    Abstract_Fiber*() const noexcept
      { return this->fiber;  }
  };

}  // namespace

POSEIDON_STATIC_CLASS_DEFINE(Fiber_Scheduler)
  {
    // constant data
    once_flag m_init_once;
    ::pthread_key_t m_sched_key;
    ::sem_t m_sched_sem[1];

    // configuration
    mutable simple_mutex m_conf_mutex;
    Config_Scalars m_conf;

    // dynamic data
    mutable simple_mutex m_sched_mutex;
    ::std::vector<PQ_Element> m_sched_pq;
    ::std::vector<rcptr<Abstract_Fiber>> m_sched_sleep_q;
    ::std::vector<rcptr<Abstract_Fiber>> m_sched_ready_q;

    static
    void
    do_start()
      {
        self->m_init_once.call(
          [&] {
            // Create a thread-specific key for the per-thread context.
            // Note it is never destroyed.
            ::pthread_key_t key;
            int err = ::pthread_key_create(&key, nullptr);
            if(err != 0)
              POSEIDON_THROW(
                  "failed to allocate thread-specific key for fibers\n"
                  "[`pthread_key_create()` failed: $1]",
                  format_errno(err));

            auto key_guard = ::rocket::make_unique_handle(key, ::pthread_key_delete);

            // Create the semaphore.
            // Note it is never destroyed.
            err = ::sem_init(self->m_sched_sem, 0, 0);
            if(err != 0)
              POSEIDON_THROW(
                  "failed to initialize semaphore\n"
                  "[`sem_init()` failed: $1]",
                  format_errno(err));

            auto sem_guard = ::rocket::make_unique_handle(self->m_sched_sem, ::sem_destroy);

            // Set up initialized data.
            simple_mutex::unique_lock lock(self->m_sched_mutex);
            self->m_sched_key = key_guard.release();
            sem_guard.release();
          });
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
    mut_thread_context()
      {
        auto ptr = ::pthread_getspecific(self->m_sched_key);
        if(ROCKET_EXPECT(ptr))
          return static_cast<Thread_Context*>(ptr);

        // Allocate it if one hasn't been allocated yet.
        auto qctx = ::rocket::make_unique<Thread_Context>();

        int err = ::pthread_setspecific(self->m_sched_key, qctx);
        if(err != 0)
          POSEIDON_THROW(
              "could not set fiber scheduler thread context\n"
              "[`pthread_setspecific()` failed: $1]",
              format_errno(err));

        POSEIDON_LOG_TRACE("Created new fiber scheduler thread context `$1`", qctx);
        return qctx.release();
      }

    [[noreturn]] static
    void
    do_execute_fiber(int word_0, int word_1) noexcept
      {
        auto myctx = self->mut_thread_context();
        do_finish_switch_fiber(myctx);

        // Get the fiber pointer back.
        Fiber_pointer fcptr(word_0, word_1);
        Abstract_Fiber* fiber = fcptr;

        // Execute the fiber.
        ROCKET_ASSERT(fiber->state() == async_state_suspended);
        fiber->m_state.store(async_state_running);
        fiber->do_on_start();
        POSEIDON_LOG_TRACE("Starting execution of fiber `$1`", fiber);

        try {
          fiber->do_execute();
        }
        catch(exception& stdex) {
          POSEIDON_LOG_WARN("$1\n[inside fiber class `$2`]", stdex, typeid(*fiber));
        }

        ROCKET_ASSERT(fiber->state() == async_state_running);
        fiber->m_state.store(async_state_finished);
        fiber->do_on_finish();
        POSEIDON_LOG_TRACE("Finished execution of fiber `$1`", fiber);

        // Note the scheduler thread may have changed.
        myctx = self->mut_thread_context();
        do_start_switch_fiber(myctx, myctx->return_uctx);
        ::setcontext(myctx->return_uctx);
        ::std::terminate();
      }

    static
    void
    do_initialize_context(Abstract_Fiber* fiber, unique_stack&& stack) noexcept
      {
        // Initialize the user-context.
        int err = ::getcontext(fiber->m_sched_uctx);
        ROCKET_ASSERT(err == 0);

        fiber->m_sched_uctx->uc_link = reinterpret_cast<::ucontext_t*>(-0x21520FF3);
        fiber->m_sched_uctx->uc_stack = stack.release();

        // Fill in the executor function, whose argument is a copy of `fiber`.
        Fiber_pointer fcptr(fiber);
        ::makecontext(fiber->m_sched_uctx, reinterpret_cast<void (*)()>(do_execute_fiber), 2, fcptr.words[0], fcptr.words[1]);
      }

    static
    void
    do_thread_loop(void* param)
      {
        const auto& exit_sig = *(const atomic_signal*) param;
        const auto myctx = self->mut_thread_context();

        // Allow signals in the current thread. Errors are ignored.
        ::sigset_t sigset;
        ::sigemptyset(&sigset);
        ::sigaddset(&sigset, SIGINT);
        ::sigaddset(&sigset, SIGTERM);
        ::sigaddset(&sigset, SIGHUP);
        ::sigaddset(&sigset, SIGALRM);
        ::pthread_sigmask(SIG_UNBLOCK, &sigset, nullptr);

        // Reload configuration.
        simple_mutex::unique_lock lock(self->m_conf_mutex);
        const auto conf = self->m_conf;
        lock.unlock();

        // Await a fiber and pop it.
        lock.lock(self->m_sched_mutex);
        self->m_sched_sleep_q.reserve(self->m_sched_sleep_q.size() + 1);
        self->m_sched_ready_q.reserve(self->m_sched_ready_q.size() + 1);

        ::timespec ts;
        ::clock_gettime(CLOCK_MONOTONIC, &ts);
        int sig = exit_sig.load();

        while(self->m_sched_sleep_q.size()) {
          // Move a fiber from the sleep queue into the scheduler queue.
          // Note this shall guarantee strong exception safety.
          self->m_sched_pq.reserve(self->m_sched_pq.size() + 1);

          auto fiber = ::std::move(self->m_sched_sleep_q.back());
          self->m_sched_sleep_q.pop_back();
          if(fiber->m_sched_running)
            continue;

          POSEIDON_LOG_TRACE("Collected fiber `$1` from sleep queue", fiber);
          auto& elem = self->m_sched_pq.emplace_back();
          int64_t timeout = ::rocket::min(fiber->m_sched_yield_timeout, conf.fail_timeout);
          elem.time = ::rocket::min(ts.tv_sec + conf.warn_timeout, fiber->m_sched_yield_since + timeout);
          elem.serial = ++ fiber->m_sched_serial;
          elem.fiber = ::std::move(fiber);
          ::std::push_heap(self->m_sched_pq.begin(), self->m_sched_pq.end(), pq_compare);
        }

        while(self->m_sched_ready_q.size()) {
          // Move a fiber from the ready queue into the scheduler queue.
          // Note this shall guarantee strong exception safety.
          self->m_sched_pq.reserve(self->m_sched_pq.size() + 1);

          auto fiber = ::std::move(self->m_sched_ready_q.back());
          self->m_sched_ready_q.pop_back();
          if(fiber->m_sched_running)
            continue;

          POSEIDON_LOG_TRACE("Collected fiber `$1` from ready queue", fiber);
          auto& elem = self->m_sched_pq.emplace_back();
          elem.time = ts.tv_sec;
          elem.serial = ++ fiber->m_sched_serial;
          elem.fiber = ::std::move(fiber);
          ::std::push_heap(self->m_sched_pq.begin(), self->m_sched_pq.end(), pq_compare);
        }

        if(sig && self->m_sched_pq.empty()) {
          // Exit if a signal has been received and there are no more fibers.
          // Note the scheduler mutex is locked so it is safe to call `strsignal()`.
          POSEIDON_LOG_INFO("Shutting down due to signal $1: $2", sig, ::strsignal(sig));
          Async_Logger::synchronize();
          ::std::quick_exit(0);
        }

        if(sig == 0) {
          // Wait for a fiber that should be scheduled.
          if(self->m_sched_pq.empty()) {
            lock.unlock();

            ::sem_wait(self->m_sched_sem);
            return;
          }

          // Calculate the amount of time to wait.
          int64_t duration = self->m_sched_pq.front().time - ts.tv_sec;
          if(duration > 0) {
            lock.unlock();

            ::clock_gettime(CLOCK_REALTIME, &ts);
            ts.tv_sec += duration;
            ::sem_timedwait(self->m_sched_sem, &ts);
            return;
          }
        }

        // Pop the first fiber.
        ::std::pop_heap(self->m_sched_pq.begin(), self->m_sched_pq.end(), pq_compare);
        auto elem = ::std::move(self->m_sched_pq.back());
        self->m_sched_pq.pop_back();
        if(elem.fiber->m_sched_serial != elem.serial) {
          POSEIDON_LOG_TRACE("Fiber element invalidated: $1 != $2", elem.fiber->m_sched_serial, elem.serial);
          return;
        }
        ROCKET_ASSERT(elem.fiber->m_sched_running == false);
        elem.fiber->m_sched_running = true;
        lock.unlock();

        if(sig != 0) {
          // Note cancellation is only possible before initialization.
          // If the fiber stack is in use, it cannot be deallocated without
          // possibility of resource leaks.
          POSEIDON_LOG_DEBUG("Killed fiber because of signal $2: $1", elem.fiber, sig);
          if(elem.fiber->state() == async_state_pending)
            return;
        }
        else if(elem.fiber->m_zombie.load()) {
          // Note cancellation is only possible before initialization.
          // If the fiber stack is in use, it cannot be deallocated without
          // possibility of resource leaks.
          POSEIDON_LOG_DEBUG("Shut down fiber: $1", elem.fiber);
          if(elem.fiber->state() == async_state_pending)
            return;
        }
        else if(elem.fiber.unique() && !elem.fiber->m_resident.load()) {
          // Note cancellation is only possible before initialization.
          // If the fiber stack is in use, it cannot be deallocated without
          // possibility of resource leaks.
          POSEIDON_LOG_DEBUG("Killed orphan fiber: $1", elem.fiber);
          if(elem.fiber->state() == async_state_pending)
            return;
        }
        else if(elem.fiber->m_sched_futp && elem.fiber->m_sched_futp->do_is_empty()) {
          // Check for blocking conditions.
          // Note that `Promise::set_value()` first attempts to lock the future
          // before constructing the value. Only after the construction succeeds,
          // does it call `Fiber_Scheduler::signal()`.
          int64_t delta = ts.tv_sec - elem.fiber->m_sched_yield_since;
          int64_t timeout = ::rocket::min(elem.fiber->m_sched_yield_timeout, conf.fail_timeout);
          if(delta < timeout) {
            // Print a warning message if the fiber has been suspended for too long.
            if(delta >= conf.warn_timeout)
              POSEIDON_LOG_WARN(
                  "Fiber `$1` has been suspended for `$2` seconds.\n"
                  "[fiber class `$3`]",
                  elem.fiber, delta, typeid(*(elem.fiber)));

            // Put the fiber back into the queue.
            elem.time = ::rocket::min(ts.tv_sec + conf.warn_timeout, elem.fiber->m_sched_yield_since + timeout);

            lock.lock(self->m_sched_mutex);
            elem.fiber->m_sched_running = false;
            self->m_sched_pq.emplace_back(elem);
            ::std::push_heap(self->m_sched_pq.begin(), self->m_sched_pq.end(), pq_compare);
            return;
          }

          // Proceed anyway.
          if(delta >= conf.fail_timeout)
            POSEIDON_LOG_ERROR(
                "Suspension of fiber `$1` has exceeded `$2` seconds.\n"
                "This circumstance looks permanent. Please check for deadlocks.\n",
                "[fiber class `$3`]",
                elem.fiber, conf.fail_timeout, typeid(*(elem.fiber)));
        }

        if(elem.fiber->state() == async_state_pending) {
          try {
            // Perform initialization that might throw exceptions here.
            self->do_initialize_context(elem.fiber, do_allocate_stack(conf.stack_vm_size));
          }
          catch(exception& stdex) {
            POSEIDON_LOG_ERROR(
                "Failed to initialize fiber: $1\n"
                "[fiber class `$2`]",
                stdex, typeid(*(elem.fiber)));

            // Put the fiber back into the sleep queue.
            lock.lock(self->m_sched_mutex);
            elem.fiber->m_sched_running = false;
            self->m_sched_sleep_q.emplace_back(::std::move(elem.fiber));
            return;
          }

          // Note this is the only place where the state is not updated by the
          // fiber itself.
          elem.fiber->m_state.store(async_state_suspended);
        }

        ROCKET_ASSERT(elem.fiber->state() == async_state_suspended);
        myctx->current = elem.fiber;
        POSEIDON_LOG_TRACE("Resuming execution of fiber `$1`", elem.fiber);

        // Resume this fiber...
        do_start_switch_fiber(myctx, elem.fiber->m_sched_uctx);
        int err = ::swapcontext(myctx->return_uctx, elem.fiber->m_sched_uctx);
        ROCKET_ASSERT(err == 0);
        do_finish_switch_fiber(myctx);

        // ... and return here.
        myctx->current = nullptr;
        POSEIDON_LOG_TRACE("Suspended execution of fiber `$1`", elem.fiber);

        if(elem.fiber->state() == async_state_suspended) {
          // Put the fiber back into the sleep queue.
          lock.lock(self->m_sched_mutex);
          elem.fiber->m_sched_running = false;
          self->m_sched_sleep_q.emplace_back(::std::move(elem.fiber));
          return;
        }

        // Free its stack and delete it thereafter.
        ROCKET_ASSERT(elem.fiber->state() == async_state_finished);
        unique_stack stack(elem.fiber->m_sched_uctx->uc_stack);
      }
  };

void
Fiber_Scheduler::
modal_loop(const atomic_signal& exit_sig)
  {
    self->do_start();

    // Schedule fibers and block until `exit_sig` becomes non-zero.
    for(;;)
      self->do_thread_loop((void*)&exit_sig);
  }

void
Fiber_Scheduler::
reload()
  {
    // Load fiber settings into temporary objects.
    const auto file = Main_Config::copy();
    Config_Scalars conf;

    auto qint = file.get_int64_opt({"fiber","stack_vm_size"});
    if(qint) {
      // Clamp the stack size between 256KiB and 256MiB for safety.
      // The upper bound (256MiB) is a hard limit because we encode the number of
      // 64KiB chunks inside the pointer itself. Therefore, we can have at most 4096
      // chunks of 64KiB, which makes up 256MiB in total.
      int64_t rint = ::rocket::clamp(*qint, 0x4'0000, 0x1000'0000);
      if(rint != *qint)
        POSEIDON_LOG_WARN(
            "Config value `fiber.stack_vm_size` truncated to `$1`\n"
            "[value `$2` out of range]",
            rint, *qint);

      conf.stack_vm_size = static_cast<size_t>(rint);
    }
    else {
      // Get system thread stack size.
      ::rlimit rlim;
      if(::getrlimit(RLIMIT_STACK, &rlim) != 0)
        POSEIDON_THROW(
            "could not get thread stack size\n"
            "[`getrlimit()` failed: $1]",
            format_errno());

      conf.stack_vm_size = static_cast<size_t>(rlim.rlim_cur);
    }
    do_validate_stack_vm_size(conf.stack_vm_size);

    // Note a negative value indicates an infinite timeout.
    qint = file.get_int64_opt({"fiber","warn_timeout"});
    if(qint)
      conf.warn_timeout = *qint & INT64_MAX;

    qint = file.get_int64_opt({"fiber","fail_timeout"});
    if(qint)
      conf.fail_timeout = *qint & INT64_MAX;

    // During destruction of temporary objects the mutex should have been unlocked.
    // The swap operation is presumed to be fast, so we don't hold the mutex
    // for too long.
    simple_mutex::unique_lock lock(self->m_conf_mutex);
    self->m_conf = conf;
  }

Abstract_Fiber*
Fiber_Scheduler::
current_opt() noexcept
  {
    auto myctx = self->get_thread_context();
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
yield(rcptr<Abstract_Future> futp_opt, int64_t msecs)
  {
    auto myctx = self->get_thread_context();
    if(!myctx)
      POSEIDON_THROW("invalid call to `yield()` inside a non-scheduler thread");

    auto fiber = myctx->current;
    if(!fiber)
      POSEIDON_THROW("invalid call to `yield()` outside a fiber");

    // Suspend the current fiber...
    ROCKET_ASSERT(fiber->state() == async_state_running);
    fiber->m_state.store(async_state_suspended);
    fiber->do_on_suspend();
    POSEIDON_LOG_TRACE("Suspending execution of fiber `$1`", fiber);

    ::timespec ts;
    ::clock_gettime(CLOCK_MONOTONIC, &ts);

    simple_mutex::unique_lock lock(self->m_sched_mutex);
    fiber->m_sched_yield_since = ts.tv_sec;
    fiber->m_sched_yield_timeout = (int64_t) ((double) msecs * 0.001 + 0.999);

    if(futp_opt && futp_opt->do_is_empty()) {
      // The value in the future object may still be under construction, but the lock
      // here prevents the other thread from modifying the scheduler queue. We still
      // attach the fiber to the future's wait queue, which may be moved into the
      // ready queue once the other thread locks the scheduler mutex successfully.
      futp_opt->m_sched_sleep_q.emplace_back(fiber);
      fiber->m_sched_futp = futp_opt;
    }
    else {
      // Attach the fiber to the ready queue of the current thread otherwise.
      fiber->m_sched_futp = nullptr;
      ::sem_post(self->m_sched_sem);
      self->m_sched_ready_q.emplace_back(fiber);
    }
    lock.unlock();

    // Suspend this fiber...
    do_start_switch_fiber(myctx, myctx->return_uctx);
    int err = ::swapcontext(fiber->m_sched_uctx, myctx->return_uctx);
    ROCKET_ASSERT(err == 0);
    myctx = self->mut_thread_context();  // (scheduler thread may have changed)
    ROCKET_ASSERT(fiber == myctx->current);
    do_finish_switch_fiber(myctx);

    if(fiber->m_sched_futp) {
      // If the fiber has been attached to the wait queue of `*futp_opt`, it shall
      // be detached now. But note it may have been detached by another thread.
      lock.lock(self->m_sched_mutex);
      ROCKET_ASSERT(fiber->m_sched_futp == futp_opt);
      fiber->m_sched_futp = nullptr;

      auto bpos = futp_opt->m_sched_sleep_q.mut_begin();
      auto epos = futp_opt->m_sched_sleep_q.mut_end();
      while(bpos != epos) {
        if(*bpos == fiber) {
          --epos;
          ::std::iter_swap(bpos, epos);
          futp_opt->m_sched_sleep_q.erase(epos);
          break;
        }
        ++bpos;
      }
    }
    lock.unlock();

    // ... and resume here.
    ROCKET_ASSERT(myctx->current == fiber);
    ROCKET_ASSERT(fiber->state() == async_state_suspended);
    fiber->m_state.store(async_state_running);
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
      POSEIDON_THROW("null fiber pointer not valid");

    if(!fiber.unique())
      POSEIDON_THROW("fiber pointer must be unique");

    // Perform some initialization. No locking is needed here.
    fiber->m_sched_serial = 12345;
    fiber->m_sched_running = false;
    fiber->m_sched_yield_since = 0;
    fiber->m_state.store(async_state_pending);

    // Attach this fiber to the ready queue.
    simple_mutex::unique_lock lock(self->m_sched_mutex);
    ::sem_post(self->m_sched_sem);
    self->m_sched_ready_q.emplace_back(fiber);
    return fiber;
  }

bool
Fiber_Scheduler::
signal(Abstract_Future& futr) noexcept
  {
    // Detach all fibers from the future, if any.
    simple_mutex::unique_lock lock(self->m_sched_mutex);
    if(futr.m_sched_sleep_q.empty())
      return false;

    try {
      // Move all fibers from the sleep queue to the ready queue.
      ::sem_post(self->m_sched_sem);
      self->m_sched_ready_q.insert(self->m_sched_ready_q.end(), futr.m_sched_sleep_q.move_begin(), futr.m_sched_sleep_q.move_end());
    }
    catch(::std::exception& stdex) {
      // Errors are ignored, as fibers will time out eventually.
      POSEIDON_LOG_WARN("Failed to reschedule fibers: $1", stdex);
    }
    futr.m_sched_sleep_q.clear();
    return true;
  }

}  // namespace poseidon
