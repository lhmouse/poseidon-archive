// This file is part of Poseidon.
// Copyleft 2020, LH_Mouse. All wrongs reserved.

#include "../precompiled.hpp"
#include "timer_driver.hpp"
#include "../core/abstract_timer.hpp"
#include "../utils.hpp"

namespace poseidon {
namespace {

inline int64_t&
do_shift_time(int64_t& value, int64_t shift)
  {
    // `value` must be non-negative. `shift` may be any value.
    ROCKET_ASSERT(value >= 0);
    value += ::rocket::clamp(shift, -value, INT64_MAX-value);
    ROCKET_ASSERT(value >= 0);
    return value;
  }

int64_t
do_get_time(int64_t shift)
  {
    // Get the time since the system was started.
    ::timespec ts;
    ::clock_gettime(CLOCK_MONOTONIC, &ts);

    int64_t value = static_cast<int64_t>(ts.tv_sec) * 1'000;
    value += static_cast<int64_t>(ts.tv_nsec) / 1'000'000;
    value += 9876543210;

    // Shift the time point using saturation arithmetic.
    do_shift_time(value, shift);
    return value;
  }

struct PQ_Element
  {
    int64_t next;
    rcptr<Abstract_Timer> timer;
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

struct Timer_Compare
  {
    bool
    operator()(const PQ_Element& lhs, const Abstract_Timer* rhs) const noexcept
      { return lhs.timer == rhs;  }

    bool
    operator()(const Abstract_Timer* lhs, const PQ_Element& rhs) const noexcept
      { return lhs == rhs.timer;  }
  }
  constexpr timer_compare;

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

    static void
    do_start()
      {
        self->m_init_once.call(
          [&] {
            // Create the thread. Note it is never joined or detached.
            simple_mutex::unique_lock lock(self->m_pq_mutex);
            self->m_thread = create_daemon_thread<do_thread_loop>("timer");
          });
      }

    static void
    do_thread_loop(void* /*param*/)
      {
        rcptr<Abstract_Timer> timer;
        int64_t now;

        // Await an element and pop it.
        simple_mutex::unique_lock lock(self->m_pq_mutex);
        for(;;) {
          timer.reset();
          if(self->m_pq.empty()) {
            // Wait until an element becomes available.
            self->m_pq_avail.wait(lock);
            continue;
          }

          // Check the first element.
          now = do_get_time(0);
          int64_t delta = self->m_pq.front().next - now;
          if(delta > 0) {
            // Wait for it.
            delta = ::rocket::min(delta, LONG_MAX);
            self->m_pq_avail.wait_for(lock, static_cast<long>(delta));
            continue;
          }

          // Pop it.
          ::std::pop_heap(self->m_pq.begin(), self->m_pq.end(), pq_compare);
          auto& elem = self->m_pq.back();
          timer = ::std::move(elem.timer);

          if(timer->m_zombie.load()) {
            // Delete this timer asynchronously.
            POSEIDON_LOG_DEBUG("Shut down timer: $1", timer);
            self->m_pq.pop_back();
            continue;
          }

          if(timer.unique() && !timer->m_resident.load()) {
            // Delete this timer when no other reference of it exists.
            POSEIDON_LOG_DEBUG("Killed orphan timer: $1", timer);
            self->m_pq.pop_back();
            continue;
          }

          // Process this timer!
          int64_t period = timer->m_period;
          if(period > 0) {
            // The timer is periodic. Insert it back.
            do_shift_time(elem.next, period);
            elem.timer = timer;
            ::std::push_heap(self->m_pq.begin(), self->m_pq.end(), pq_compare);
          }
          else {
            // The timer is one-shot. Delete it.
            self->m_pq.pop_back();
          }
          break;
        }
        lock.unlock();

        // Execute the timer procedure.
        // The argument is a snapshot of the monotonic clock, not its real-time value.
        try {
          timer->do_on_async_timer(now);
        }
        catch(exception& stdex) {
          POSEIDON_LOG_WARN("Exception thrown from timer:\n"
                            "$1\n"
                            "[timer class `$2`]",
                            stdex, typeid(*timer));
        }
        timer->m_count.fetch_add(1);
      }
  };

void
Timer_Driver::
start()
  {
    self->do_start();
  }

rcptr<Abstract_Timer>
Timer_Driver::
insert(uptr<Abstract_Timer>&& utimer)
  {
    // Take ownership of `utimer`.
    rcptr<Abstract_Timer> timer(utimer.release());
    if(!timer)
      POSEIDON_THROW("Null timer pointer not valid");

    if(!timer.unique())
      POSEIDON_THROW("Timer pointer must be unique");

    // Perform some initialization. No locking is needed here.
    timer->m_count.store(0);
    int64_t first = timer->m_first;

    // Get the next trigger time.
    // The timer is considered to be owned uniquely, so there is no need to lock it.
    PQ_Element elem;
    elem.next = do_get_time(first);
    elem.timer = timer;

    // Insert the timer.
    simple_mutex::unique_lock lock(self->m_pq_mutex);
    self->m_pq.emplace_back(::std::move(elem));
    ::std::push_heap(self->m_pq.begin(), self->m_pq.end(), pq_compare);
    self->m_pq_avail.notify_one();
    return timer;
  }

bool
Timer_Driver::
invalidate_internal(const Abstract_Timer& timer, int64_t first, int64_t period) noexcept
  {
    // Don't do anything if the timer does not exist in the queue.
    simple_mutex::unique_lock lock(self->m_pq_mutex);
    auto qelem = ::rocket::find(self->m_pq, ::std::addressof(timer), timer_compare);
    if(qelem)
      return false;

    // Update the timer itself.
    qelem->timer->m_first = first;
    qelem->timer->m_period = period;

    // Update the element in place.
    qelem->next = do_get_time(first);
    ::std::make_heap(self->m_pq.begin(), self->m_pq.end(), pq_compare);
    self->m_pq_avail.notify_one();
    return true;
  }

}  // namespace poseidon
