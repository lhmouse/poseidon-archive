// This file is part of Poseidon.
// Copyleft 2022 - 2023, LH_Mouse. All wrongs reserved.

#include "../precompiled.ipp"
#include "easy_timer.hpp"
#include "../base/abstract_timer.hpp"
#include "../static/timer_driver.hpp"
#include "../fiber/abstract_fiber.hpp"
#include "../static/fiber_scheduler.hpp"
#include "../static/async_logger.hpp"
#include "../utils.hpp"
#include <time.h>

namespace poseidon {
namespace {

using thunk_function = void (void*, int64_t);

struct Final_Fiber final : Abstract_Fiber
  {
    thunk_function* m_cb_thunk;
    weak_ptr<void> m_cb_wobj;

    int64_t m_now;

    explicit
    Final_Fiber(thunk_function* cb_thunk, shared_ptrR<void> cb_obj, int64_t now)
      : m_cb_thunk(cb_thunk), m_cb_wobj(cb_obj), m_now(now)
      { }

    virtual
    void
    do_abstract_fiber_on_work() override
      {
        auto cb_obj = this->m_cb_wobj.lock();
        if(!cb_obj)
          return;

        // We are in the main thread, so invoke the user-defined callback here.
        this->m_cb_thunk(cb_obj.get(), this->m_now);
      }
  };

struct Final_Timer final : Abstract_Timer
  {
    thunk_function* m_cb_thunk;
    weak_ptr<void> m_cb_wobj;

    explicit
    Final_Timer(thunk_function* cb_thunk, shared_ptrR<void> cb_obj)
      : m_cb_thunk(cb_thunk), m_cb_wobj(cb_obj)
      { }

    virtual
    void
    do_abstract_timer_on_tick(int64_t now)
      {
        auto cb_obj = this->m_cb_wobj.lock();
        if(!cb_obj)
          return;

        // We are in the timer thread here, so create a new fiber.
        fiber_scheduler.insert(::std::make_unique<Final_Fiber>(this->m_cb_thunk, cb_obj, now));
      }
  };

}  // namespace

Easy_Timer::
~Easy_Timer()
  {
  }

void
Easy_Timer::
start(int64_t delay, int64_t period)
  {
    if(!this->m_timer)
      this->m_timer = ::std::make_shared<Final_Timer>(this->m_cb_thunk, this->m_cb_obj);

    timer_driver.insert(this->m_timer, delay, period);
  }

void
Easy_Timer::
stop() noexcept
  {
    this->m_timer = nullptr;
  }

}  // namespace poseidon
