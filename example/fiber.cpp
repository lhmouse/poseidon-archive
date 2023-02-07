// This file is part of Poseidon.
// Copyleft 2022, LH_Mouse. All wrongs reserved.

#include "../poseidon/precompiled.ipp"
#include "../poseidon/fiber/abstract_fiber.hpp"
#include "../poseidon/fiber/future.hpp"
#include "../poseidon/base/abstract_timer.hpp"
#include "../poseidon/static/fiber_scheduler.hpp"
#include "../poseidon/static/timer_driver.hpp"
#include "../poseidon/static/async_logger.hpp"
#include "../poseidon/utils.hpp"

namespace {
using namespace poseidon;

struct Future_Timer : Abstract_Timer
  {
    int64_t value;
    shared_ptr<future<int>> futr;

    explicit
    Future_Timer(int64_t x)
      : value(x), futr(::std::make_shared<future<int>>())
      {
        POSEIDON_LOG_FATAL(("new timer `$1`"), this);
      }

    ~Future_Timer()
      {
        POSEIDON_LOG_FATAL(("delete timer `$1`"), this);
      }

    void
    do_abstract_timer_on_tick(int64_t /*now*/) override
      {
        POSEIDON_LOG_FATAL(("timer sets value: $1"), this->value);
        this->futr->set_value(this->value);
      }
  };

struct Example_Fiber : Abstract_Fiber
  {
    int value;

    explicit
    Example_Fiber(int seconds)
      : value(seconds)
      {
        POSEIDON_LOG_ERROR(("new fiber `$1`: $2"), this, this->value);
      }

    ~Example_Fiber() override
      {
        POSEIDON_LOG_ERROR(("delete fiber `$1`: $2"), this, this->value);
      }

    void
    do_abstract_fiber_on_execution() override
      {
        POSEIDON_LOG_WARN(("fiber `$1`: init"), this);

        auto timer = ::std::make_shared<Future_Timer>(this->value);
        timer_driver.insert(timer, this->value * 1000000000LL, 0);  // delay, period
        this->yield(timer->futr);
        POSEIDON_LOG_WARN(("fiber `$1`: first pass: value = $2"), this, timer->futr->value());

        timer = ::std::make_shared<Future_Timer>(this->value);
        timer_driver.insert(timer, (this->value + 6) * 1000000000LL, 0);  // delay, period
        this->yield(timer->futr);
        POSEIDON_LOG_WARN(("fiber `$1`: second pass: value = $2"), this, timer->futr->value());
      }
  };

struct Init
  {
    Init()
      {
        for(int k = 1;  k <= 5;  ++k)
          fiber_scheduler.insert(::std::make_unique<Example_Fiber>(k));
      }
  }
  const init;

}  // namespace
