// This file is part of Poseidon.
// Copyleft 2020, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_CORE_PROMISE_
#define POSEIDON_CORE_PROMISE_

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
    explicit
    Promise() noexcept
      { }

    Promise(Promise&& other) noexcept
      : m_futp(::std::move(other.m_futp))
      { }

    Promise&
    operator=(Promise&& other) noexcept
      {
        this->~Promise();
        ::rocket::construct(this, ::std::move(other));
        return *this;
      }

  public:
    ~Promise();

    // Gets a pointer to the associated future.
    futp<ValueT>
    future()
      {
        if(!this->m_futp)
          this->m_futp = ::rocket::make_refcnt<Future<ValueT>>();

        // Return a shared future.
        return this->m_futp;
      }

    // Puts a value into the associated future.
    // If the future is not empty, `false` is returned, and there is no effect.
    template<typename... ParamsT,
    ROCKET_DISABLE_IF(::std::is_void<ValueT>::value && sizeof...(ParamsT))>
    bool
    set_value(ParamsT&&... params)
      {
        if(!this->m_futp)
          ::rocket::sprintf_and_throw<::std::invalid_argument>(
                "Promise: no future associated (value type `%s`)",
                typeid(ValueT).name());

        // Construct a new value in the future.
        bool new_value_set = false;

        this->m_futp->m_once.call(
          [&] {
            ROCKET_ASSERT(this->m_futp->m_state.load() == future_state_empty);
            ::rocket::construct(&(this->m_futp->m_value), ::std::forward<ParamsT>(params)...);
            this->m_futp->m_state.store(future_state_value);
            new_value_set = true;
          });

        if(new_value_set)
          Fiber_Scheduler::signal(*(this->m_futp));

        return new_value_set;
      }

    // Puts an exception into the associated future.
    // If the future is not empty, `false` is returned, and there is no effect.
    bool
    set_exception(const ::std::exception_ptr& eptr)
      {
        if(!eptr)
          ::rocket::sprintf_and_throw<::std::invalid_argument>(
                "Promise: null exception pointer (value type `%s`)",
                typeid(ValueT).name());

        if(!this->m_futp)
          ::rocket::sprintf_and_throw<::std::invalid_argument>(
                "Promise: no future associated (value type `%s`)",
                typeid(ValueT).name());

        // Construct a new exception pointer in the future.
        bool new_value_set = false;

        this->m_futp->m_once.call(
          [&] {
            ROCKET_ASSERT(this->m_futp->m_state.load() == future_state_empty);
            ::rocket::construct(&(this->m_futp->m_exptr), eptr);
            this->m_futp->m_state.store(future_state_except);
            new_value_set = true;
          });

        if(new_value_set)
          Fiber_Scheduler::signal(*(this->m_futp));

        return new_value_set;
      }

    bool
    set_current_exception()
      {
        return this->set_exception(::std::current_exception());
      }
  };

template<typename ValueT>
Promise<ValueT>::
~Promise()
  {
    if(!this->m_futp)
      return;

    // Construct a null exception pointer in the future.
    bool new_value_set = false;

    this->m_futp->m_once.call(
      [&] {
        ROCKET_ASSERT(this->m_futp->m_state.load() == future_state_empty);
        ::rocket::construct(&(this->m_futp->m_exptr));
        this->m_futp->m_state.store(future_state_except);
      });

    if(new_value_set)
      Fiber_Scheduler::signal(*(this->m_futp));
  }

}  // namespace poseidon

#endif
