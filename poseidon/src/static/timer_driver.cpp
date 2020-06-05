// This file is part of Poseidon.
// Copyleft 2020, LH_Mouse. All wrongs reserved.

#include "../precompiled.hpp"
#include "timer_driver.hpp"
#include "../core/abstract_timer.hpp"
#include "../xutilities.hpp"

namespace poseidon {
namespace {

inline
int64_t&
do_shift_time(int64_t& value, int64_t shift)
noexcept
  {
    // `value` must be non-negative. `shift` may be any value.
    ROCKET_ASSERT(value >= 0);
    value += ::rocket::clamp(shift, -value, INT64_MAX-value);
    ROCKET_ASSERT(value >= 0);
    return value;
  }

int64_t
do_get_time(int64_t shift)
noexcept
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
    constexpr
    bool
    operator()(const PQ_Element& lhs, const PQ_Element& rhs)
    const noexcept
      { return lhs.next > rhs.next;  }

    constexpr
    bool
    operator()(const PQ_Element& lhs, int64_t rhs)
    const noexcept
      { return lhs.next > rhs;  }

    constexpr
    bool
    operator()(int64_t lhs, const PQ_Element& rhs)
    const noexcept
      { return lhs > rhs.next;  }
  }
  constexpr pq_compare;

}  // namespace

POSEIDON_STATIC_CLASS_DEFINE(Timer_Driver)
  {
    // constant data
    ::pthread_t m_thread;

    // dynamic data
    mutable Si_Mutex m_pq_mutex;
    Cond_Var m_pq_avail;
    ::std::vector<PQ_Element> m_pq;
  };

void
Timer_Driver::
do_thread_loop(void* /*param*/)
  {
    rcptr<Abstract_Timer> timer;
    int64_t now;

    // Await an element and pop it.
    Si_Mutex::unique_lock lock(self->m_pq_mutex);
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
        self->m_pq_avail.wait_for(lock,
                    static_cast<long>(::rocket::min(delta, LONG_MAX)));
        continue;
      }

      // Pop it.
      ::std::pop_heap(self->m_pq.begin(), self->m_pq.end(), pq_compare);
      timer = ::std::move(self->m_pq.back().timer);
      if(!timer.unique()) {
        // Process this timer!
        int64_t period = timer->m_period.load(::std::memory_order_relaxed);
        if(period > 0) {
          // The timer is periodic. Insert it back.
          self->m_pq.back().timer = timer;
          do_shift_time(self->m_pq.back().next, period);
          ::std::push_heap(self->m_pq.begin(), self->m_pq.end(), pq_compare);
        }
        else {
          // The timer is one-shot. Delete it.
          self->m_pq.pop_back();
        }
        break;
      }

      // Delete this timer when no other reference of it exists.
      POSEIDON_LOG_DEBUG("Killed orphan timer: $1", timer);
      self->m_pq.pop_back();
    }
    lock.unlock();

    // Execute the timer procedure.
    // The argument is a snapshot of the monotonic clock, not its real-time value.
    try {
      timer->do_on_async_timer(now);
    }
    catch(exception& stdex) {
      POSEIDON_LOG_WARN("Exception thrown from timer: $1\n"
                        "[timer class `$2`]",
                        stdex.what(), typeid(*timer).name());
    }
    timer->m_count.fetch_add(1, ::std::memory_order_release);
  }

void
Timer_Driver::
start()
  {
    if(self->m_thread)
      return;

    // Create the thread. Note it is never joined or detached.
    Si_Mutex::unique_lock lock(self->m_pq_mutex);
    self->m_thread = create_daemon_thread<do_thread_loop>("timer");
  }

rcptr<Abstract_Timer>
Timer_Driver::
insert(uptr<Abstract_Timer>&& utimer)
  {
    // Take ownership of `utimer`.
    rcptr<Abstract_Timer> timer(utimer.release());
    if(!timer)
      POSEIDON_THROW("null timer pointer not valid");

    if(!timer.unique())
      POSEIDON_THROW("timer pointer must be unique");

    // Get the next trigger time.
    // The timer is considered to be owned uniquely, so there is no need to lock it.
    int64_t next = do_get_time(timer->m_first);

    // Lock priority queue for modification.
    Si_Mutex::unique_lock lock(self->m_pq_mutex);

    // Insert the timer.
    self->m_pq.push_back({ next, timer });
    ::std::push_heap(self->m_pq.begin(), self->m_pq.end(), pq_compare);
    self->m_pq_avail.notify_one();
    return timer;
  }

bool
Timer_Driver::
invalidate_internal(const Abstract_Timer* ctimer)
noexcept
  {
    // Lock priority queue for modification.
    Si_Mutex::unique_lock lock(self->m_pq_mutex);

    // Don't do anything if the timer does not exist in the queue.
    PQ_Element* qelem;
    if(::std::none_of(self->m_pq.begin(), self->m_pq.end(),
                      [&](PQ_Element& r) { return (qelem = &r)->timer.get() == ctimer;  }))
      return false;

    // Update the element in place.
    qelem->next = do_get_time(ctimer->m_first.load(::std::memory_order_relaxed));
    ::std::make_heap(self->m_pq.begin(), self->m_pq.end(), pq_compare);
    self->m_pq_avail.notify_one();
    return true;
  }

}  // namespace poseidon
