// This file is part of Poseidon.
// Copyleft 2020, LH_Mouse. All wrongs reserved.

#include "../src/precompiled.hpp"
#include "../src/core/abstract_fiber.hpp"
#include "../src/static/fiber_scheduler.hpp"
#include "../src/core/abstract_timer.hpp"
#include "../src/static/timer_driver.hpp"
#include "../src/utils.hpp"

namespace {
using namespace poseidon;

struct Promise_Timer : Abstract_Timer
  {
    Promise<int> prom;
    int64_t value;

    explicit
    Promise_Timer(int64_t seconds)
      : Abstract_Timer(0, 1000),
        value(seconds)
      {
//        POSEIDON_LOG_FATAL("new timer `$1`", this);

        // create the future
        this->prom.future();
      }

    ~Promise_Timer()
      override
      {
//        POSEIDON_LOG_FATAL("delete timer `$1`", this);
      }

    void
    do_on_async_timer(int64_t /*now*/)
      override
      {
        POSEIDON_LOG_FATAL("set promise: $1", this->value);
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
//        POSEIDON_LOG_ERROR("new fiber `$1`: $2", this, this->value);
      }

    ~Example_Fiber()
      override
      {
//        POSEIDON_LOG_ERROR("delete fiber `$1`: $2", this, this->value);
      }

    void
    do_execute()
      {
        POSEIDON_LOG_WARN("fiber `$1`: init", this);

        for(;;)
          try {
            auto timer = Timer_Driver::insert(::rocket::make_unique<Promise_Timer>(0));
            auto futr = ::rocket::static_pointer_cast<Promise_Timer>(timer)->prom.future();
            Fiber_Scheduler::yield(futr);
            POSEIDON_LOG_WARN("fiber `$1`: value = $2", this, futr->value());
          }
          catch(::std::exception& e) {
            POSEIDON_LOG_ERROR("error: $1", e);
          }
      }
  };

const auto plain =
  {
    Fiber_Scheduler::insert(::rocket::make_unique<Example_Fiber>(1)),
    Fiber_Scheduler::insert(::rocket::make_unique<Example_Fiber>(2)),
    Fiber_Scheduler::insert(::rocket::make_unique<Example_Fiber>(3)),
  };

const auto resident =
  {
    (Fiber_Scheduler::insert(::rocket::make_unique<Example_Fiber>(1))->set_resident(), 1),
    (Fiber_Scheduler::insert(::rocket::make_unique<Example_Fiber>(2))->set_resident(), 2),
    (Fiber_Scheduler::insert(::rocket::make_unique<Example_Fiber>(3))->set_resident(), 3),
    (Fiber_Scheduler::insert(::rocket::make_unique<Example_Fiber>(4))->set_resident(), 4),
    (Fiber_Scheduler::insert(::rocket::make_unique<Example_Fiber>(5))->set_resident(), 5),
    (Fiber_Scheduler::insert(::rocket::make_unique<Example_Fiber>(6))->set_resident(), 6),
    (Fiber_Scheduler::insert(::rocket::make_unique<Example_Fiber>(7))->set_resident(), 7),
    (Fiber_Scheduler::insert(::rocket::make_unique<Example_Fiber>(8))->set_resident(), 8),
    (Fiber_Scheduler::insert(::rocket::make_unique<Example_Fiber>(9))->set_resident(), 9),
  };

}  // namespace
