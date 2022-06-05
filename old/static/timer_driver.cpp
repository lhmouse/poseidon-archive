// This file is part of Poseidon.
// Copyleft 2020, LH_Mouse. All wrongs reserved.

#include "../precompiled.ipp"
#include "timer_driver.hpp"
#include "async_logger.hpp"
#include "../core/abstract_timer.hpp"
#include "../utils.hpp"
#include <time.h>
#include <signal.h>

namespace poseidon {
namespace {

inline
int64_t
do_shift_time(int64_t& value, int64_t shift) noexcept
  {
    // `value` must be non-negative. `shift` may be any value.
    ROCKET_ASSERT(value >= 0);
    int64_t res;
    value = ROCKET_ADD_OVERFLOW(value, shift, &res) ? INT64_MAX : (res & (~res >> 63));
    return value;
  }

int64_t
do_get_time(int64_t& value, int64_t shift) noexcept
  {
    // Get the time since the system was started.
    ::timespec ts;
    ::clock_gettime(CLOCK_MONOTONIC, &ts);
    value = (int64_t) ts.tv_sec * 1000 + (int64_t) ts.tv_nsec / 1000000;
    return do_shift_time(value, shift);
  }

struct PQ_Element
  {
    int64_t next;
    weak_ptr<Abstract_Timer> timer;
  };

struct PQ_Compare
  {
    bool
    operator()(const PQ_Element& lhs, const PQ_Element& rhs) const noexcept
      { return lhs.next > rhs.next;  }

    bool
    operator()(const PQ_Element& lhs, int64_t rhs) const noexcept
      { return lhs.next > rhs;  }

    bool
    operator()(int64_t lhs, const PQ_Element& rhs) const noexcept
      { return lhs > rhs.next;  }
  }
  constexpr pq_compare;

}  // namespace

POSEIDON_STATIC_CLASS_DEFINE(Timer_Driver)
  {
    // constant data
    once_flag m_init_once;
    ::pthread_t m_thread;

    // dynamic data
    mutable simple_mutex m_pq_mutex;
    condition_variable m_pq_avail;
    ::std::vector<PQ_Element> m_pq;

    [[noreturn]] static
    void*
    do_thread_procedure(void*)
      {
        // Set thread information. Errors are ignored.
        ::sigset_t sigset;
        ::sigemptyset(&sigset);
        ::sigaddset(&sigset, SIGINT);
        ::sigaddset(&sigset, SIGTERM);
        ::sigaddset(&sigset, SIGHUP);
        ::sigaddset(&sigset, SIGALRM);
        ::pthread_sigmask(SIG_BLOCK, &sigset, nullptr);

        int oldst;
        ::pthread_setcancelstate(PTHREAD_CANCEL_DISABLE, &oldst);

        ::pthread_setname_np(::pthread_self(), "timer");

        // Enter an infinite loop.
        for(;;)
          try {
            self->do_thread_loop();
          }
          catch(exception& stdex) {
            POSEIDON_LOG_FATAL(
                "Caught an exception from timer thread loop: $1\n"
                "[exception class `$2`]\n",
                stdex.what(), typeid(stdex).name());
          }
      }

    static
    void
    do_thread_loop()
      {
        // Await a timer.
        simple_mutex::unique_lock lock(self->m_pq_mutex);
        int64_t now;
        while(self->m_pq.empty() || (do_get_time(now, 0) < self->m_pq.front().next))
          if(self->m_pq.empty())
            self->m_pq_avail.wait(lock);
          else
            self->m_pq_avail.wait_for(lock, self->m_pq.front().next - now);

        ::std::pop_heap(self->m_pq.begin(), self->m_pq.end(), pq_compare);
        auto elem = ::std::move(self->m_pq.back());
        self->m_pq.pop_back();
        lock.unlock();

        auto timer = elem.timer.lock();
        if(!timer)
          return;
        else if(timer->m_cancelled.load())
          return;

        if(timer->m_period != 0) {
          // If the timer is periodic, insert it back.
          do_shift_time(elem.next, timer->m_period);

          lock.lock(self->m_pq_mutex);
          self->m_pq.emplace_back(elem);
          ::std::push_heap(self->m_pq.begin(), self->m_pq.end(), pq_compare);
          lock.unlock();
        }

        // Execute the timer procedure. The argument is a snapshot of the
        // monotonic clock, not its real-time value.
        POSEIDON_LOG_TRACE("Starting execution of timer `$1`: now = $2", timer.get(), now);
        timer->m_count.xadd(1);

        try {
          timer->do_abstract_timer_interval(now);
        }
        catch(exception& stdex) {
          POSEIDON_LOG_WARN(
              "Exception thrown from timer:\n"
              "$1\n"
              "[timer class `$2`]",
              stdex, typeid(timer));
        }
      }
  };

shared_ptr<Abstract_Timer>
Timer_Driver::
insert(const shared_ptr<Abstract_Timer>& timer)
  {
    // Perform daemon initialization.
    self->m_init_once.call(
      [] {
        POSEIDON_LOG_INFO("Initializing timer driver...");
        simple_mutex::unique_lock lock(self->m_pq_mutex);

        // Create the thread. Note it is never joined or detached.
        int err = ::pthread_create(&(self->m_thread), nullptr, self->do_thread_procedure, nullptr);
        if(err != 0) ::std::terminate();
      });

    if(!timer)
      POSEIDON_THROW("null timer pointer not valid");

    if(!timer.unique())
      POSEIDON_THROW("timer pointer must be unique");

    // Perform some initialization. No locking is needed here.
    timer->m_count.store(0);
    int64_t first = timer->m_first;

    // Get the next trigger time.
    // The timer is considered to be owned uniquely, so there is no need to lock it.
    PQ_Element elem;
    do_get_time(elem.next, first);
    elem.timer = timer;

    // Insert the timer.
    simple_mutex::unique_lock lock(self->m_pq_mutex);
    self->m_pq.emplace_back(::std::move(elem));
    ::std::push_heap(self->m_pq.begin(), self->m_pq.end(), pq_compare);
    self->m_pq_avail.notify_one();
    return timer;
  }

}  // namespace poseidon
