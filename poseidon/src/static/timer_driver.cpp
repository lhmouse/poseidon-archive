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

ROCKET_PURE_FUNCTION
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
    ::pthread_t m_thread;

    struct
      {
        mutable Si_Mutex mutex;
        Cond_Var avail;
        ::std::vector<PQ_Element> heap;
      }
      m_queue;
  };

void
Timer_Driver::
do_thread_loop(void* /*param*/)
  {
    // Await an element and pop it.
    Si_Mutex::unique_lock lock(self->m_queue.mutex);
    int64_t now;
    for(;;) {
      if(self->m_queue.heap.size()) {
        // Check the minimum element.
        now = do_get_time(0);
        long delta = static_cast<long>(::rocket::clamp(
                           self->m_queue.heap.front().next - now, 0, LONG_MAX));
        if(delta == 0)
          break;
        self->m_queue.avail.wait_for(lock, delta);
      }
      else
        // Wait until an element becomes available.
        self->m_queue.avail.wait(lock);
    }
    ::std::pop_heap(self->m_queue.heap.begin(), self->m_queue.heap.end(), pq_compare);

    // Process this timer!
    auto timer = self->m_queue.heap.back().timer;
    if(timer->use_count() == 2) {
      // Delete this timer when no other reference of it exists.
      self->m_queue.heap.pop_back();
      return;
    }

    // Get the next trigger time.
    Si_Mutex::unique_lock tlock(timer->m_mutex);
    if(timer->m_period > 0) {
      // Update the element in place.
      do_shift_time(self->m_queue.heap.back().next, timer->m_period);
      ::std::push_heap(self->m_queue.heap.begin(), self->m_queue.heap.end(), pq_compare);
    }
    else {
      // Delete this one-shot timer.
      self->m_queue.heap.pop_back();
    }
    tlock.unlock();

    // Leave critical section.
    lock.unlock();

    // Execute the timer procedure.
    // The argument is a snapshot of the monotonic clock, not its real-time value.
    timer->do_on_async_timer(now);
  }

void
Timer_Driver::
start()
  {
    if(self->m_thread)
      return;

    // Create the thread. Note it is never joined or detached.
    auto thr = create_daemon_thread<do_thread_loop>("timer");
    self->m_thread = ::std::move(thr);
  }

rcptr<Abstract_Timer>
Timer_Driver::
insert(uptr<Abstract_Timer>&& utimer)
  {
    // Take ownership of `timer`.
    rcptr<Abstract_Timer> timer(utimer.release());
    if(!timer)
      POSEIDON_THROW("null timer pointer not valid");

    // Get the next trigger time.
    // The timer is considered to be owned uniquely, so there is no need to lock it.
    auto next = do_get_time(timer->m_first);

    // Lock priority queue for modification.
    Si_Mutex::unique_lock lock(self->m_queue.mutex);

    // Insert the timer.
    self->m_queue.heap.push_back({ next, timer });
    ::std::push_heap(self->m_queue.heap.begin(), self->m_queue.heap.end(), pq_compare);
    self->m_queue.avail.notify_one();
    return timer;
  }

bool
Timer_Driver::
invalidate_internal(Abstract_Timer* timer)
noexcept
  {
    // Lock priority queue for modification.
    Si_Mutex::unique_lock lock(self->m_queue.mutex);

    // Don't do anything if the timer does not exist in the queue.
    auto qelem = ::std::find_if(self->m_queue.heap.begin(), self->m_queue.heap.end(),
                     [&](const PQ_Element& elem) { return elem.timer.get() == timer;  });
    if(qelem == self->m_queue.heap.end())
      return false;

    // Get the next trigger time.
    Si_Mutex::unique_lock tlock(timer->m_mutex);
    auto next = do_get_time(timer->m_first);
    tlock.unlock();

    // Update the element in place.
    qelem->next = next;
    ::std::make_heap(self->m_queue.heap.begin(), self->m_queue.heap.end(), pq_compare);
    self->m_queue.avail.notify_one();
    return true;
  }

}  // namespace poseidon
