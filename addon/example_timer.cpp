// This file is part of Poseidon.
// Copyleft 2022, LH_Mouse. All wrongs reserved.

#include "../src/precompiled.ipp"
#include "../src/core/abstract_timer.hpp"
#include "../src/static/timer_driver.hpp"
#include "../src/static/async_logger.hpp"
#include "../src/utils.hpp"

namespace {
using namespace poseidon;

struct Example_Timer : Abstract_Timer
  {
    explicit
    Example_Timer()
      {
        POSEIDON_LOG_ERROR(("example timer created"));
      }

    void
    do_abstract_timer_on_tick(int64_t now) override
      {
        POSEIDON_LOG_ERROR(("example timer: now = $1, count = $2"), now, this->count());
      }
  };

shared_ptr<Example_Timer>
do_create_timer()
  {
    auto timer = ::std::make_shared<Example_Timer>();
    timer_driver.insert(timer, 3000'000000, 500'000000);
    return timer;
  }

const auto timer = do_create_timer();

}  // namespace
