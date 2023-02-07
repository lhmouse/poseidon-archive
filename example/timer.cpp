// This file is part of Poseidon.
// Copyleft 2022 - 2023, LH_Mouse. All wrongs reserved.

#include "../poseidon/precompiled.ipp"
#include "../poseidon/easy_timer.hpp"
#include "../poseidon/static/async_logger.hpp"
#include "../poseidon/utils.hpp"

namespace {
using namespace ::poseidon;

extern Easy_Timer my_timer;

void
timer_callback(int64_t now)
  {
    POSEIDON_LOG_WARN(("example timer: now = $1"), now);
  }

int
start_timer()
  {
    constexpr int64_t delay = 5'000'000'000;  // 5 seconds
    constexpr int64_t period = 1'000'000'000;  // 1 seconds

    my_timer.start(delay, period);
    POSEIDON_LOG_ERROR(("example timer started: delay = $1, period = $2"), delay, period);
    return 0;
  }

// Start the timer when this shared object is being loaded.
Easy_Timer my_timer(timer_callback);
int dummy = start_timer();

}  // namespace
