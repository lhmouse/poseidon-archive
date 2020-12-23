// This file is part of Poseidon.
// Copyleft 2020, LH_Mouse. All wrongs reserved.

#include "../precompiled.hpp"
#include "fiber_scheduler.hpp"
#include "main_config.hpp"
#include "../core/abstract_fiber.hpp"
#include "../core/config_file.hpp"
#include "../utils.hpp"
#include <sys/resource.h>
#include <sys/mman.h>
#include <semaphore.h>
#include <signal.h>

#ifdef POSEIDON_ENABLE_ADDRESS_SANITIZER
extern "C" {

void
__sanitizer_start_switch_fiber(void** save, const void* sp_base, size_t st_size)
  noexcept;

void
__sanitizer_finish_switch_fiber(void* save, const void** sp_base, size_t* st_size)
  noexcept;

#define POSEIDON_ASAN_START_SWITCH_FIBER(myctx, uctx)  \
    (::__sanitizer_start_switch_fiber(&((myctx)->asan_fiber_save),  \
                               (uctx)->uc_stack.ss_sp, (uctx)->uc_stack.ss_size))

#define POSEIDON_ASAN_FINISH_SWITCH_FIBER(myctx)  \
    (::__sanitizer_finish_switch_fiber((myctx)->asan_fiber_save, nullptr, nullptr))

}  // extern "C"

#else  // POSEIDON_ENABLE_ADDRESS_SANITIZER

#define POSEIDON_ASAN_START_SWITCH_FIBER(myctx, uctx)   ((void)0)
#define POSEIDON_ASAN_FINISH_SWITCH_FIBER(myctx)        ((void)0)

#endif  // POSEIDON_ENABLE_ADDRESS_SANITIZER

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

struct Stack_pointer
  {
    void* base;
    size_t size;

    constexpr
    Stack_pointer(nullptr_t = nullptr)
      noexcept
      : base(nullptr), size(0)
      { }

    constexpr
    Stack_pointer(const ::stack_t& st)
      : base(st.ss_sp), size(st.ss_size)
      { }

    constexpr
    Stack_pointer(void* xbase, size_t xsize)
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

const size_t s_page_size = static_cast<size_t>(::sysconf(_SC_PAGESIZE));
simple_mutex s_stack_pool_mutex;
Stack_pointer s_stack_pool_head;

size_t
do_validate_stack_vm_size(size_t stack_vm_size)
  {
    if(stack_vm_size & 0xFFFF)
      POSEIDON_THROW("Stack size `$1` not a multiple of 64KiB", stack_vm_size);

    uintptr_t msize = (stack_vm_size >> 16) - 1;
    if(msize > 0xFFF)
      POSEIDON_THROW("Stack size `$1` out of range", stack_vm_size);

    if(stack_vm_size < s_page_size * 4)
      POSEIDON_THROW("Stack size `$1` less than 4 pages", stack_vm_size);

    return stack_vm_size;
  }

void
do_unmap_stack_aux(Stack_pointer sp)
  noexcept
  {
    char* vm_base = static_cast<char*>(sp.base) - s_page_size;
    size_t vm_size = sp.size + s_page_size * 2;

    // Note that on Linux `munmap()` may fail with `ENOMEM`.
    // There is little we can do so we ignore this error.
    if(::munmap(vm_base, vm_size) != 0)
      POSEIDON_LOG_FATAL("Could not deallocate virtual memory (base `$2`, size `$3`)\n"
                         "[`munmap()` failed: $1]",
                         format_errno(errno), vm_base, vm_size);
  }

struct Stack_delete
  {
    constexpr
    Stack_pointer
    null()
      const noexcept
      { return nullptr;  }

    constexpr
    bool
    is_null(Stack_pointer sp)
      const noexcept
      { return sp.base == nullptr;  }

    void
    close(Stack_pointer sp)
      noexcept
      {
        simple_mutex::unique_lock lock(s_stack_pool_mutex);

        // Insert the region at the beginning.
        auto qnext = static_cast<Stack_pointer*>(sp.base);
        qnext = ::rocket::construct_at(qnext, s_stack_pool_head);
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
      ::rocket::destroy_at(qnext);
      lock.unlock();

      // Use this region if it is large enough.
      if(ROCKET_EXPECT(sp.size + s_page_size * 2 >= vm_size))
        return unique_stack(sp);

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
                     format_errno(errno), vm_size);

    sp.base = vm_base + s_page_size;
    sp.size = vm_size - s_page_size * 2;
    auto sp_guard = ::rocket::make_unique_handle(sp, do_unmap_stack_aux);

    // Mark stack area writable.
    if(::mprotect(sp.base, sp.size, PROT_READ | PROT_WRITE) != 0)
      POSEIDON_THROW("Could not set stack memory permission (base `$2`, size `$3`)\n"
                     "[`mprotect()` failed: $1]",
                     format_errno(errno), sp.base, sp.size);

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
    void* asan_fiber_save;  // used by address sanitizer
    ::ucontext_t return_uctx[1];
  };

union Fancy_Fiber_Pointer
  {
    Abstract_Fiber* fiber;
    int words[2];

    explicit constexpr
    Fancy_Fiber_Pointer(Abstract_Fiber* xfiber)
      noexcept
      : fiber(xfiber)
      { }

    explicit constexpr
    Fancy_Fiber_Pointer(int word_0, int word_1)
      noexcept
      : words{ word_0, word_1 }
      { }

    constexpr operator
    Abstract_Fiber*()
      const noexcept
      { return this->fiber;  }
  };

class Queue_Semaphore
  {
  private:
    ::sem_t m_sem[1];

  public:
    explicit
    Queue_Semaphore()
      {
        int r = ::sem_init(this->m_sem, 0, 0);
        if(r != 0)
          POSEIDON_THROW("Failed to initialize semaphore\n"
                         "[`sem_init()` failed: $1]",
                         format_errno(errno));
      }

    ASTERIA_NONCOPYABLE_DESTRUCTOR(Queue_Semaphore)
      {
        int r = ::sem_destroy(this->m_sem);
        ROCKET_ASSERT(r == 0);
      }

  public:
    bool
    wait_for(long secs)
      noexcept
      {
        int r;
        if(secs <= 0) {
          // Deal with immediate timeouts.
          r = ::sem_trywait(this->m_sem);
        }
        else {
          // Get the current time.
          ::timespec ts;
          r = ::clock_gettime(CLOCK_REALTIME, &ts);
          ROCKET_ASSERT(r == 0);

          // Ensure we don't cause overflows.
          if(secs <= ::std::numeric_limits<::time_t>::max() - ts.tv_sec) {
            ts.tv_sec += static_cast<::time_t>(secs);
            r = ::sem_timedwait(this->m_sem, &ts);
          }
          else
            r = ::sem_wait(this->m_sem);
        }
        ROCKET_ASSERT(r != EINVAL);
        return r == 0;
      }

    bool
    wait()
      noexcept
      {
        // Note `semwait()` may fail with `EINTR`.
        int r = ::sem_wait(this->m_sem);
        ROCKET_ASSERT(r != EINVAL);
        return r == 0;
      }

    bool
    signal()
      noexcept
      {
        // Note `sem_post()` may fail with `EOVERFLOW`.
        // The semaphore only indicates whether the queue is non-empty,
        // so we may ignore this error.
        int r = ::sem_post(this->m_sem);
        ROCKET_ASSERT(r != EINVAL);
        return r == 0;
      }
  };

}  // namespace

POSEIDON_STATIC_CLASS_DEFINE(Fiber_Scheduler)
  {
    // constant data
    once_flag m_init_once;
    ::pthread_key_t m_sched_key;

    // configuration
    mutable simple_mutex m_conf_mutex;
    Config_Scalars m_conf;

    // dynamic data
    mutable simple_mutex m_sched_mutex;
    Queue_Semaphore m_sched_avail;
    ::std::vector<PQ_Element> m_sched_pq;
    Abstract_Fiber* m_sched_sleep_head = nullptr;
    Abstract_Fiber* m_sched_ready_head = nullptr;

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
              POSEIDON_THROW("Failed to allocate thread-specific key for fibers\n"
                             "[`pthread_key_create()` failed: $1]",
                             format_errno(err));

            auto key_guard = ::rocket::make_unique_handle(key, ::pthread_key_delete);

            // Set up initialized data.
            simple_mutex::unique_lock lock(self->m_sched_mutex);
            self->m_sched_key = key_guard.release();
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
    open_thread_context()
      {
        auto ptr = ::pthread_getspecific(self->m_sched_key);
        if(ROCKET_EXPECT(ptr))
          return static_cast<Thread_Context*>(ptr);

        // Allocate it if one hasn't been allocated yet.
        auto qctx = ::rocket::make_unique<Thread_Context>();

        int err = ::pthread_setspecific(self->m_sched_key, qctx);
        if(err != 0)
          POSEIDON_THROW("Could not set fiber scheduler thread context\n"
                         "[`pthread_setspecific()` failed: $1]",
                         format_errno(err));

        POSEIDON_LOG_TRACE("Created new fiber scheduler thread context `$1`", qctx);
        return qctx.release();
      }

    [[noreturn]] static
    void
    do_execute_fiber(int word_0, int word_1)
      noexcept
      {
        auto myctx = self->open_thread_context();
        POSEIDON_ASAN_FINISH_SWITCH_FIBER(myctx);

        // Get the fiber pointer back.
        Fancy_Fiber_Pointer fcptr(word_0, word_1);
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
        myctx = self->open_thread_context();
        POSEIDON_ASAN_START_SWITCH_FIBER(myctx, myctx->return_uctx);
        ::setcontext(myctx->return_uctx);
        ::std::terminate();
      }

    static
    void
    do_initialize_context(Abstract_Fiber* fiber, unique_stack&& stack)
      noexcept
      {
        // Initialize the user-context.
        int r = ::getcontext(fiber->m_sched_uctx);
        ROCKET_ASSERT(r == 0);

        fiber->m_sched_uctx->uc_link = reinterpret_cast<::ucontext_t*>(-0x21520FF3);
        fiber->m_sched_uctx->uc_stack = stack.release();

        // Fill in the executor function, whose argument is a copy of `fiber`.
        Fancy_Fiber_Pointer fcptr(fiber);
        ::makecontext(fiber->m_sched_uctx, reinterpret_cast<void (*)()>(do_execute_fiber),
                                           2, fcptr.words[0], fcptr.words[1]);
      }

    static
    void
    do_thread_loop(void* param)
      {
        const auto& exit_sig = *(const atomic_signal*)param;
        const auto myctx = self->open_thread_context();

        rcptr<Abstract_Fiber> fiber;
        int64_t now;
        unique_stack stack;

        // Reload configuration.
        simple_mutex::unique_lock lock(self->m_conf_mutex);
        const auto conf = self->m_conf;
        lock.unlock();

        // Await a fiber and pop it.
        lock.lock(self->m_sched_mutex);
        for(;;) {
          fiber.reset();
          now = do_get_monotonic_seconds();
          int sig = exit_sig.load();

          while(self->m_sched_sleep_head) {
            // Move a fiber from the sleep queue into the scheduler queue.
            // Note this shall guarantee strong exception safety.
            self->m_sched_pq.emplace_back();
            fiber.reset(self->m_sched_sleep_head);
            self->m_sched_sleep_head = fiber->m_sched_sleep_next;

            // An odd version number indicates the fiber is being scheduled.
            if(fiber->m_sched_version % 2) {
              self->m_sched_pq.pop_back();
              continue;
            }
            POSEIDON_LOG_TRACE("Collected fiber `$1` from sleep queue", fiber);
            fiber->m_sched_version += 2;

            auto& elem = self->m_sched_pq.back();
            int64_t timeout = ::rocket::min(fiber->m_sched_yield_timeout, conf.fail_timeout);
            elem.time = ::rocket::min(now + conf.warn_timeout,
                                      fiber->m_sched_yield_since + timeout);
            elem.version = fiber->m_sched_version;
            elem.fiber = ::std::move(fiber);
            ::std::push_heap(self->m_sched_pq.begin(), self->m_sched_pq.end(), pq_compare);
          }

          while(self->m_sched_ready_head) {
            // Move a fiber from the ready queue into the scheduler queue.
            // Note this shall guarantee strong exception safety.
            self->m_sched_pq.emplace_back();
            fiber.reset(self->m_sched_ready_head);
            self->m_sched_ready_head = fiber->m_sched_ready_next;

            // An odd version number indicates the fiber is being scheduled.
            if(fiber->m_sched_version % 2) {
              self->m_sched_pq.pop_back();
              continue;
            }
            POSEIDON_LOG_TRACE("Collected fiber `$1` from ready queue", fiber);
            fiber->m_sched_version += 2;

            auto& elem = self->m_sched_pq.back();
            elem.time = now;
            elem.version = fiber->m_sched_version;
            elem.fiber = ::std::move(fiber);
            ::std::push_heap(self->m_sched_pq.begin(), self->m_sched_pq.end(), pq_compare);
          }

          if(sig && self->m_sched_pq.empty()) {
            // Exit if a signal has been received and there are no more fibers.
            // Note the scheduler mutex is locked so it is safe to call `strsignal()`.
            POSEIDON_LOG_INFO("Shutting down due to signal $1: $2", sig, ::strsignal(sig));
            Async_Logger::synchronize(1000);
            ::std::quick_exit(0);
          }

          // Calculate the time to sleep.
          long time_wait;
          if(sig)
            time_wait = 0;  // don't wait at all
          else if(self->m_sched_pq.empty())
            time_wait = LONG_MAX;  // wait infinitely
          else
            time_wait = self->m_sched_pq.front().time - now;

          if(time_wait > 0) {
            lock.unlock();
            self->m_sched_avail.wait_for(time_wait);
            lock.lock(self->m_sched_mutex);
            continue;
          }

          // Pop the first fiber.
          ::std::pop_heap(self->m_sched_pq.begin(), self->m_sched_pq.end(), pq_compare);
          auto& elem = self->m_sched_pq.back();
          fiber = ::std::move(elem.fiber);

          ROCKET_ASSERT(elem.version % 2 == 0);
          if(fiber->m_sched_version != elem.version) {
            self->m_sched_pq.pop_back();  // invalidated
            continue;
          }

          bool should_cancel = sig ||   // termination pending
                               fiber->m_zombie.load() ||  // shut down explicitly
                               (fiber.unique() && !fiber->m_resident.load());  // orphan

          if(should_cancel && (fiber->state() == async_state_pending)) {
            // Note cancellation is only possible before initialization.
            // If the fiber stack is in use, it cannot be deallocated without possibility
            // of resource leaks.
            POSEIDON_LOG_INFO("Killed fiber `$1`\n[fiber class `$2`]", fiber, typeid(*fiber));
            self->m_sched_pq.pop_back();
            continue;
          }

          // Check for blocking conditions.
          // Note that `Promise::set_value()` first attempts to lock the future, then
          // constructs the value. Only after the construction succeeds, does it call
          // `Fiber_Scheduler::signal()`.
          if(!should_cancel && fiber->m_sched_futp && fiber->m_sched_futp->do_is_empty()) {
            // Check wait duration.
            int64_t delta = now - fiber->m_sched_yield_since;
            int64_t timeout = ::rocket::min(fiber->m_sched_yield_timeout, conf.fail_timeout);
            if(delta < timeout) {
              // Print a warning message if the fiber has been suspended for too long.
              if(delta >= conf.warn_timeout)
                POSEIDON_LOG_WARN("Fiber `$1` has been suspended for `$2` seconds.\n",
                                  "[fiber class `$3`]",
                                  fiber, delta, typeid(*fiber));

              // Put the fiber back into the queue.
              elem.time = ::rocket::min(now + conf.warn_timeout,
                                        fiber->m_sched_yield_since + timeout);
              elem.fiber = ::std::move(fiber);
              ::std::push_heap(self->m_sched_pq.begin(), self->m_sched_pq.end(), pq_compare);
              continue;
            }

            // Proceed anyway.
            if(delta >= conf.fail_timeout)
              POSEIDON_LOG_ERROR("Suspension of fiber `$1` has exceeded `$2` seconds.\n"
                        "This circumstance looks permanent. Please check for deadlocks.\n",
                        "[fiber class `$3`]",
                        fiber, conf.fail_timeout, typeid(*fiber));
          }

          // Process this fiber!
          // An odd version number indicates the fiber is being scheduled.
          ROCKET_ASSERT(fiber->m_sched_version % 2 == 0);
          fiber->m_sched_version += 1;
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
            POSEIDON_LOG_ERROR("Failed to initialize fiber: $1\n"
                               "[fiber class `$2`]",
                               stdex, typeid(*fiber));

            // Put the fiber back into the sleep queue.
            lock.lock(self->m_sched_mutex);
            ROCKET_ASSERT(fiber->m_sched_version % 2 != 0);
            fiber->m_sched_version += 1;
            fiber->m_sched_sleep_next = ::std::exchange(self->m_sched_sleep_head, fiber);
            fiber.release();
            return;
          }

          // Note this shall not throw exceptions.
          self->do_initialize_context(fiber, ::std::move(stack));

          // Finish initialization.
          // Note this is the only scenerio where the fiber state is not updated
          // by itself.
          fiber->m_state.store(async_state_suspended);
        }

        ROCKET_ASSERT(fiber->state() == async_state_suspended);
        myctx->current = fiber;
        POSEIDON_LOG_TRACE("Resuming execution of fiber `$1`", fiber);

        // Resume this fiber...
        POSEIDON_ASAN_START_SWITCH_FIBER(myctx, fiber->m_sched_uctx);
        int r = ::swapcontext(myctx->return_uctx, fiber->m_sched_uctx);
        ROCKET_ASSERT(r == 0);
        POSEIDON_ASAN_FINISH_SWITCH_FIBER(myctx);

        // ... and return here.
        myctx->current = nullptr;
        POSEIDON_LOG_TRACE("Suspended execution of fiber `$1`", fiber);

        if(fiber->state() == async_state_suspended) {
          // Put the fiber back into the sleep queue.
          lock.lock(self->m_sched_mutex);
          ROCKET_ASSERT(fiber->m_sched_version % 2 != 0);
          fiber->m_sched_version += 1;
          fiber->m_sched_sleep_next = ::std::exchange(self->m_sched_sleep_head, fiber);
          fiber.release();
          return;
        }

        // Otherwise, the fiber shall have completed execution.
        // Free its stack. The fiber can be safely deleted thereafter.
        ROCKET_ASSERT(fiber->state() == async_state_finished);
        stack.reset(fiber->m_sched_uctx->uc_stack);
      }

    static
    void
    do_signal_if_queues_empty()
      noexcept
      {
        if(ROCKET_EXPECT(self->m_sched_sleep_head))
          return;

        if(ROCKET_EXPECT(self->m_sched_ready_head))
          return;

        self->m_sched_avail.signal();
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
      // The upper bound (256MiB) is a hard limit, because we encode the number of
      // 64KiB chunks inside the pointer itself, so we can have at most 4096 64KiB
      // pages, which makes up 256MiB in total.
      int64_t rint = ::rocket::clamp(*qint, 0x4'0000, 0x1000'0000);
      if(rint != *qint)
        POSEIDON_LOG_WARN("Config value `fiber.stack_vm_size` truncated to `$1`\n"
                          "[value `$2` out of range]",
                          rint, *qint);

      conf.stack_vm_size = static_cast<size_t>(rint);
    }
    else {
      // Get system thread stack size.
      ::rlimit rlim;
      if(::getrlimit(RLIMIT_STACK, &rlim) != 0)
        POSEIDON_THROW("Could not get thread stack size\n"
                       "[`getrlimit()` failed: $1]",
                       format_errno(errno));

      conf.stack_vm_size = static_cast<size_t>(rlim.rlim_cur);
    }
    do_validate_stack_vm_size(conf.stack_vm_size);

    // Note a negative value indicates an infinite timeout.
    qint = file.get_int64_opt({"fiber","warn_timeout"});
    if(qint)
      conf.warn_timeout = (*qint | *qint >> 63) & INT64_MAX;

    qint = file.get_int64_opt({"fiber","fail_timeout"});
    if(qint)
      conf.fail_timeout = (*qint | *qint >> 63) & INT64_MAX;

    // During destruction of temporary objects the mutex should have been unlocked.
    // The swap operation is presumed to be fast, so we don't hold the mutex
    // for too long.
    simple_mutex::unique_lock lock(self->m_conf_mutex);
    self->m_conf = conf;
  }

Abstract_Fiber*
Fiber_Scheduler::
current_opt()
  noexcept
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
yield(rcptr<const Abstract_Future> futp_opt, int64_t msecs)
  {
    auto myctx = self->get_thread_context();
    if(!myctx)
      POSEIDON_THROW("Invalid call to `yield()` inside a non-scheduler thread");

    auto fiber = myctx->current;
    if(!fiber)
      POSEIDON_THROW("Invalid call to `yield()` outside a fiber");

    // Suspend the current fiber...
    ROCKET_ASSERT(fiber->state() == async_state_running);
    fiber->m_state.store(async_state_suspended);
    fiber->do_on_suspend();
    POSEIDON_LOG_TRACE("Suspending execution of fiber `$1`", fiber);

    int64_t now = do_get_monotonic_seconds();
    int64_t timeout = ::rocket::clamp(msecs, 0, INT64_MAX - 999) + 999;
    timeout = static_cast<int64_t>(static_cast<uint64_t>(timeout) / 1000);

    simple_mutex::unique_lock lock(self->m_sched_mutex);
    fiber->m_sched_yield_since = now;
    fiber->m_sched_yield_timeout = timeout;

    if(futp_opt && futp_opt->do_is_empty()) {
      // The value in the future object may be still under construction, but the lock
      // here prevents the other thread from modifying the scheduler queue. We still
      // attach the fiber to the future's wait queue, which may be moved into the
      // ready queue once the other thread locks the scheduler mutex successfully.
      fiber->m_sched_futp = futp_opt.get();
      fiber->m_sched_ready_next = ::std::exchange(futp_opt->m_sched_waiting_head, fiber);
    }
    else {
      // Attach the fiber to the ready queue of the current thread otherwise.
      fiber->m_sched_futp = nullptr;
      self->do_signal_if_queues_empty();
      fiber->m_sched_ready_next = ::std::exchange(self->m_sched_ready_head, fiber);
    }
    fiber->add_reference();
    lock.unlock();

    // Suspend this fiber...
    POSEIDON_ASAN_START_SWITCH_FIBER(myctx, myctx->return_uctx);
    int r = ::swapcontext(fiber->m_sched_uctx, myctx->return_uctx);
    ROCKET_ASSERT(r == 0);
    myctx = self->open_thread_context();  // (scheduler thread may have changed)
    POSEIDON_ASAN_FINISH_SWITCH_FIBER(myctx);

    if(fiber->m_sched_futp) {
      // If the fiber has been attached to the wait queue of `*futp_opt`, it shall
      // be detached now, unless it has been detached by another thread of course.
      lock.lock(self->m_sched_mutex);
      ROCKET_ASSERT(fiber->m_sched_futp == futp_opt);
      fiber->m_sched_futp = nullptr;

      // Remove `fiber` from the wait queue.
      auto mref = ::std::ref(futp_opt->m_sched_waiting_head);
      while(mref && (mref != fiber))
        mref = ::std::ref(mref.get()->m_sched_ready_next);

      if(mref)
        mref.get() = fiber->m_sched_ready_next,
          fiber->drop_reference();
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
      POSEIDON_THROW("Null fiber pointer not valid");

    if(!fiber.unique())
      POSEIDON_THROW("Fiber pointer must be unique");

    // Perform some initialization. No locking is needed here.
    fiber->m_sched_version = 0;
    fiber->m_sched_yield_since = 0;
    fiber->m_sched_futp = nullptr;
    fiber->m_sched_sleep_next = nullptr;
    fiber->m_state.store(async_state_pending);

    // Attach this fiber to the ready queue.
    simple_mutex::unique_lock lock(self->m_sched_mutex);
    self->do_signal_if_queues_empty();
    fiber->m_sched_ready_next = ::std::exchange(self->m_sched_ready_head, fiber);
    fiber->add_reference();
    return fiber;
  }

bool
Fiber_Scheduler::
signal(const Abstract_Future& futr)
  noexcept
  {
    // Detach all fibers from the future, if any.
    simple_mutex::unique_lock lock(self->m_sched_mutex);
    if(!futr.m_sched_waiting_head)
      return false;

    // Splice the two queues.
    // Fibers are moved from one queue to the other, so there is no need to tamper
    // with reference counts here.
    auto mref = ::std::ref(self->m_sched_ready_head);
    while(mref)
      mref = ::std::ref(mref.get()->m_sched_ready_next);

    self->do_signal_if_queues_empty();
    mref.get() = ::std::exchange(futr.m_sched_waiting_head, nullptr);
    return true;
  }

}  // namespace poseidon
