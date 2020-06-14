// This file is part of Poseidon.
// Copyleft 2020, LH_Mouse. All wrongs reserved.

#include "../src/precompiled.hpp"
#include "../src/core/abstract_fiber.hpp"
#include "../src/static/fiber_scheduler.hpp"
#include "../src/core/abstract_timer.hpp"
#include "../src/static/timer_driver.hpp"
#include "../src/utilities.hpp"

namespace {

using namespace poseidon;

struct Promise_Timer : Abstract_Timer
  {
    Promise<int> prom;
    int64_t value;

    explicit
    Promise_Timer(int64_t seconds)
      : Abstract_Timer(seconds * 1000, 0),
        value(seconds)
      { }

    void
    do_on_async_timer(int64_t /*now*/)
    override
      {
        POSEIDON_LOG_DEBUG("set promise: $1", this->value);
        this->prom.set_value(this->value);
      }
  };

struct Example_Fiber : Abstract_Fiber
  {
    int64_t value;

    explicit
    Example_Fiber(int64_t seconds)
      : value(seconds)
      {
        POSEIDON_LOG_ERROR("new fiber `$1`: $2", this, this->value);
      }

    ~Example_Fiber()
    override
      {
        POSEIDON_LOG_ERROR("delete fiber `$1`: $2", this, this->value);
      }

    void
    do_execute()
      {
        POSEIDON_LOG_WARN("fiber `$1`: init", this);

        auto timer = Timer_Driver::insert(::rocket::make_unique<Promise_Timer>(this->value));
        auto futr = ::rocket::static_pointer_cast<Promise_Timer>(timer)->prom.future();
        Fiber_Scheduler::yield(futr);
        POSEIDON_LOG_WARN("fiber `$1`: value = $2", this, futr->get_value());

        timer = Timer_Driver::insert(::rocket::make_unique<Promise_Timer>(this->value + 3));
        futr = ::rocket::static_pointer_cast<Promise_Timer>(timer)->prom.future();
        Fiber_Scheduler::yield(futr);
        POSEIDON_LOG_WARN("fiber `$1`: value = $2", this, futr->get_value());
      }
  };

const auto fib1 = Fiber_Scheduler::insert(::rocket::make_unique<Example_Fiber>(1));
const auto fib2 = Fiber_Scheduler::insert(::rocket::make_unique<Example_Fiber>(2));
const auto fib3 = Fiber_Scheduler::insert(::rocket::make_unique<Example_Fiber>(3));
const auto fib4 = Fiber_Scheduler::insert(::rocket::make_unique<Example_Fiber>(1));
const auto fib5 = Fiber_Scheduler::insert(::rocket::make_unique<Example_Fiber>(2));
const auto fib6 = Fiber_Scheduler::insert(::rocket::make_unique<Example_Fiber>(3));

}  // namespace
