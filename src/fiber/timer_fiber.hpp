// This file is part of Poseidon.
// Copyleft 2022, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_FIBER_TIMER_FIBER_
#define POSEIDON_FIBER_TIMER_FIBER_

#include "../fwd.hpp"
#include "abstract_fiber.hpp"

namespace poseidon {

class Timer_Fiber
  : public Abstract_Fiber
  {
  private:
    struct Tick_Queue;
    shared_ptr<Tick_Queue> m_queue;

    struct Tick_Timer;
    shared_ptr<Tick_Timer> m_timer;

  public:
    // Creates a new fiber. The arguments are passed to `Timer_Driver::insert()`
    // to initialize the timer.
    explicit
    Timer_Fiber(Timer_Driver& driver, int64_t delay, int64_t period);

  protected:
    // This implements `Abstract_Fiber`.
    virtual
    void
    do_abstract_fiber_on_execution() override;

    // This callback is invoked by the fiber scheduler for each timer 'tick', and
    // is intended to be overriden by derived classes. `now` is the number of
    // nanoseconds since system startup and `count` is the number of ticks that
    // has been generated for this timer, both measured by the timer thread.
    virtual
    void
    do_on_timer(int64_t now, int64_t count) = 0;

  public:
    ASTERIA_NONCOPYABLE_VIRTUAL_DESTRUCTOR(Timer_Fiber);

    // Gets the timer.
    // If `stop_timre()` has been called, a null pointer is returned.
    const shared_ptr<Tick_Timer>&
    timer_opt() const noexcept
      { return this->m_timer;  }

    // Stops the timer and prevents further calls to `do_on_timer()`.
    void
    stop() noexcept;
  };

}  // namespace poseidon
#endif
