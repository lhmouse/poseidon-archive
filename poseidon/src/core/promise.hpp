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
    void
    do_set_ready_nolock()
    noexcept
      {
        auto futp = this->m_futp.get();
        ROCKET_ASSERT(futp);

        ROCKET_ASSERT(futp->m_stor.index() != future_state_empty);
        futp->m_count.store(1);
        Fiber_Scheduler::signal(*futp);
      }

    bool
    do_dispose()
    noexcept
      {
        auto futp = this->m_futp.get();
        if(!futp)
          return false;

        mutex::unique_lock lock(futp->m_mutex);
        if(futp->m_stor.index() != future_state_empty)
          return false;

        // Mark the future broken.
        futp->m_stor.template emplace<future_state_except>();
        this->do_set_ready_nolock();
        return true;
      }

    [[noreturn]]
    void
    do_throw_no_future()
    const
      {
        ::rocket::sprintf_and_throw<::std::invalid_argument>(
              "Promise: no future associated (value type `%s`)",
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
    template<typename... ArgsT,
    ROCKET_DISABLE_IF(::std::is_void<ValueT>::value && sizeof...(ArgsT))>
    bool
    set_value(ArgsT&&... args)
      {
        auto futp = this->m_futp.get();
        if(!futp)
          this->do_throw_no_future();

        mutex::unique_lock lock(futp->m_mutex);
        if(futp->m_stor.index() != future_state_empty)
          return false;

        // Construct a new value in the future.
        futp->m_stor.template emplace<future_state_value>(::std::forward<ArgsT>(args)...);
        this->do_set_ready_nolock();
        return true;
      }

    // Puts an exception into the associated future.
    // If the future is not empty, `false` is returned, and there is no effect.
    bool
    set_exception(const ::std::exception_ptr& eptr_opt)
      {
        auto futp = this->m_futp.get();
        if(!futp)
          this->do_throw_no_future();

        // Get a pointer to the exception to set.
        // If the user passes a null pointer, the current exception is used.
        auto eptr = eptr_opt;
        if(!eptr)
          eptr = ::std::current_exception();

        if(!eptr)
          ::rocket::sprintf_and_throw<::std::invalid_argument>(
            "Promise: no current exception (value type `%s`)",
            typeid(ValueT).name());

        // Check future state.
        mutex::unique_lock lock(futp->m_mutex);
        if(futp->m_stor.index() != future_state_empty)
          return false;

        // Construct a exception pointer in the future.
        futp->m_stor.template emplace<future_state_except>(::std::move(eptr));
        this->do_set_ready_nolock();
        return true;
      }
  };

}  // namespace poseidon

#endif
