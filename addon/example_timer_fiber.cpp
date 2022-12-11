// This file is part of Poseidon.
// Copyleft 2022, LH_Mouse. All wrongs reserved.

#include "../poseidon/precompiled.ipp"
#include "../poseidon/fiber/timer_fiber.hpp"
#include "../poseidon/static/fiber_scheduler.hpp"
#include "../poseidon/static/async_logger.hpp"
#include "../poseidon/utils.hpp"

namespace {
using namespace poseidon;

struct Example_Fiber : Timer_Fiber
  {
    explicit
    Example_Fiber()
      :
        Timer_Fiber(timer_driver, 5000000000LL, 1500000000LL)
      {
        POSEIDON_LOG_FATAL(("new timer fiber `$1`"), this);
      }

    ~Example_Fiber()
      {
        POSEIDON_LOG_FATAL(("delete timer fiber `$1`"), this);
      }

    virtual
    void
    do_on_timer(int64_t now, int64_t count) override
      {
        POSEIDON_LOG_ERROR(("timer fiber `$1` tick: now = $2, count = $3"), this, now, count);
        if(count < 5)
          return;

        this->stop();
        POSEIDON_LOG_ERROR(("timer fiber `$1` stopped"), this);
      }
  };

struct Init
  {
    Init()
      {
        fiber_scheduler.insert(::std::make_unique<Example_Fiber>());
      }
  }
  const init;

}  // namespace
