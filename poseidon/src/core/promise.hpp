// This file is part of Poseidon.
// Copyleft 2020, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_CORE_PROMISE_HPP_
#define POSEIDON_CORE_PROMISE_HPP_

#include "future.hpp"
#include "../static/fiber_scheduler.hpp"

namespace poseidon {

template<typename ValueT>
class Promise
  {
    friend Future<ValueT>;

  private:
    rcptr<Future<ValueT>> m_futp;

  public:
    Promise()
    noexcept
      { }

    Promise(Promise&& other)
    noexcept
      : m_futp(::std::move(other.m_futp))
      { }

    Promise&
    operator=(Promise&& other)
    noexcept
      {
        if(this->m_futp == other.m_futp)
          return *this;

        this->do_dispose();
        this->m_futp = ::std::move(other.m_futp);
        return *this;
      }

    ~Promise()
      { this->do_dispose();  }

  private:
    bool
    do_dispose()
    noexcept
      {
        auto futp = this->m_futp.get();
        if(!futp)
          return false;

        simple_mutex::unique_lock lock(futp->m_mutex);
        if(futp->m_stor.index() != future_state_empty)
          return false;

        // Mark the future broken.
        futp->m_stor.template emplace<future_state_except>();
        lock.unlock();

        Fiber_Scheduler::signal(*futp);
        return true;
      }

    [[noreturn]]
    void
    do_throw_no_future()
    const
      {
        ::rocket::sprintf_and_throw<::std::invalid_argument>(
              "Promise: No future associated (value type `%s`)",
              typeid(ValueT).name());
      }

  public:
    // Gets a reference to the associated future.
    // If `m_futp` is null, a new future is created and associated.
    futp<ValueT>
    future()
      {
        auto futp = this->m_futp;
        if(futp)
          return futp;

        // Create a new future if one hasn't been allocated.
        futp = ::rocket::make_refcnt<Future<ValueT>>();
        this->m_futp = futp;
        return futp;
      }

    // Puts a value into the associated future.
    // If the future is not empty, `false` is returned, and there is no effect.
    template<typename... ParamsT,
    ROCKET_DISABLE_IF(::std::is_void<ValueT>::value && sizeof...(ParamsT))>
    bool
    set_value(ParamsT&&... params)
      {
        auto futp = this->m_futp.get();
        if(!futp)
          this->do_throw_no_future();

        simple_mutex::unique_lock lock(futp->m_mutex);
        if(futp->m_stor.index() != future_state_empty)
          return false;

        // Construct a new value in the future.
        futp->m_stor.template emplace<future_state_value>(::std::forward<ParamsT>(params)...);
        lock.unlock();

        Fiber_Scheduler::signal(*futp);
        return true;
      }

    // Puts an exception into the associated future.
    // If the future is not empty, `false` is returned, and there is no effect.
    bool
    set_exception(const ::std::exception_ptr& eptr)
      {
        auto futp = this->m_futp.get();
        if(!futp)
          this->do_throw_no_future();

        if(!eptr)
          ::rocket::sprintf_and_throw<::std::invalid_argument>(
                "Promise: Null exception pointer (value type `%s`)",
                typeid(ValueT).name());

        // Check future state.
        simple_mutex::unique_lock lock(futp->m_mutex);
        if(futp->m_stor.index() != future_state_empty)
          return false;

        // Construct a exception pointer in the future.
        futp->m_stor.template emplace<future_state_except>(::std::move(eptr));
        lock.unlock();

        Fiber_Scheduler::signal(*futp);
        return true;
      }

    bool
    set_current_exception()
      {
        return this->set_exception(::std::current_exception());
      }
  };

}  // namespace poseidon

#endif
