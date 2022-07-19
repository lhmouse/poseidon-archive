// This file is part of Poseidon.
// Copyleft 2022, LH_Mouse. All wrongs reserved.

#include "../precompiled.ipp"
#include "timer_fiber.hpp"
#include "future.hpp"
#include "../timer/abstract_timer.hpp"
#include "../static/async_logger.hpp"
#include "../static/timer_driver.hpp"
#include "../utils.hpp"

namespace poseidon {
namespace {

struct Tick
  {
    int64_t now;
    int64_t count;
  };

struct Tick_Queue
  {
    mutable plain_mutex mutex;
    shared_ptr<Future<void>> queue_cond;
    linear_buffer queue;
  };

struct Tick_Timer : Abstract_Timer
  {
    weak_ptr<Tick_Queue> m_queue;

    virtual
    void
    do_abstract_timer_on_tick(int64_t now) override
      {
        auto queue = this->m_queue.lock();
        if(!queue)
          return;

        plain_mutex::unique_lock lock(queue->mutex);

        Tick tick;
        tick.now = now;
        tick.count = this->count();
        POSEIDON_LOG_TRACE(("Pushing new timer tick: now = $1, count = $2"), tick.now, tick.count);
        queue->queue.putn((const char*) &tick, sizeof(tick));

        if(queue->queue_cond)
          queue->queue_cond->set_value();
      }
  };

}  // namespace

POSEIDON_HIDDEN_STRUCT(Timer_Fiber, Tick_Queue);
POSEIDON_HIDDEN_STRUCT(Timer_Fiber, Tick_Timer);

Timer_Fiber::
Timer_Fiber(Timer_Driver& driver, int64_t delay, int64_t period)
  {
    this->m_queue = ::std::make_shared<Tick_Queue>();
    this->m_timer = ::std::make_shared<Tick_Timer>();
    this->m_timer->m_queue = this->m_queue;
    driver.insert(this->m_timer, delay, period);
  }

Timer_Fiber::
~Timer_Fiber()
  {
  }

void
Timer_Fiber::
do_abstract_fiber_on_execution()
  {
    while(auto queue = this->m_queue) {
      Tick tick;
      shared_ptr<Future<void>> queue_cond;

      // Wait for an element.
      plain_mutex::unique_lock lock(queue->mutex);
      if(!queue->queue.empty()) {
        // Pop a tick.
        ROCKET_ASSERT(queue->queue.size() >= sizeof(tick));
        queue->queue.getn((char*) &tick, sizeof(tick));
      }
      else if(queue->queue_cond && queue->queue_cond->empty()) {
        // Join an existent fiber in waiting.
        queue_cond = queue->queue_cond;
      }
      else {
        // Start waiting.
        queue_cond = ::std::make_shared<Future<void>>();
        queue->queue_cond = queue_cond;
      }
      lock.unlock();

      // If we should wait, do it.
      if(queue_cond) {
        this->yield(queue_cond, INT64_MAX);
        queue_cond->value();
        continue;
      }

      // Got a new element.
      this->do_on_timer(tick.now, tick.count);
    }
  }

void
Timer_Fiber::
stop() noexcept
  {
    this->m_queue.reset();
    this->m_timer.reset();
  }

}  // namespace poseidon
