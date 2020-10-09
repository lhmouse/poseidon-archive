// This file is part of Poseidon.
// Copyleft 2020, LH_Mouse. All wrongs reserved.

#include "../src/precompiled.hpp"
#include "../src/core/abstract_timer.hpp"
#include "../src/static/timer_driver.hpp"
#include "../src/util.hpp"

namespace {

using namespace poseidon;

constexpr int64_t first = 3000;  // trigger after 3000ms
constexpr int64_t period = 5000;  // repeat evert 5000ms

struct Example_Timer : Abstract_Timer
  {
    Example_Timer()
      : Abstract_Timer(first, period)
      {
        POSEIDON_LOG_ERROR("example timer created: first = $1, period = $2",
                           first, period);
      }

    void
    do_on_async_timer(int64_t now)
    override
      {
        POSEIDON_LOG_ERROR("example timer running: now = $1", now);
      }
  };

const auto timer = Timer_Driver::insert(::rocket::make_unique<Example_Timer>());

}  // namespace
