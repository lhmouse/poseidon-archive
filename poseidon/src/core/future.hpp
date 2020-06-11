// This file is part of Poseidon.
// Copyleft 2020, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_CORE_FUTURE_HPP_
#define POSEIDON_CORE_FUTURE_HPP_

#include "abstract_future.hpp"

namespace poseidon {

template<typename ValueT>
class Future
  : public ::asteria::Rcfwd<Future<ValueT>>,
    public Abstract_Future
  {
    friend Promise<ValueT>;

  private:
    ::rocket::variant<
          ::rocket::nullopt_t,      // future_state_empty
          typename ::std::conditional<
              ::std::is_void<ValueT>::value,
              int, ValueT>::type,   // future_state_value
          ::std::exception_ptr      // future_state_except
      > m_stor;

  public:
    Future()
    noexcept
      { }

    ASTERIA_NONCOPYABLE_DESTRUCTOR(Future);

  private:
    [[noreturn]]
    void
    do_throw_valueless_unlocked()
    const
      {
        // Check whether this future is empty.
        if(this->m_stor.index() == future_state_empty)
          ::rocket::sprintf_and_throw<::std::invalid_argument>(
              "Future: no value set yet (value type `%s`)",
              typeid(ValueT).name());

        // Check for pending exceptions.
        ROCKET_ASSERT(this->m_stor.index() == future_state_except);
        const auto& eptr = this->m_stor.template as<future_state_except>();

        // This state indicates either an exception has been set or the associated
        // promise went out of scope without setting a value.
        if(!eptr)
          ::rocket::sprintf_and_throw<::std::invalid_argument>(
              "Future: broken promise (value type `%s`)",
              typeid(ValueT).name());

        // Rethrow the pending exception.
        ::std::rethrow_exception(eptr);
      }

  public:
    // Gets the state, which is any of `future_state_empty`, `future_state_value`
    // or `future_state_except`.
    //
    // * `future_state_empty` indicates no value has been set yet.
    //   Any retrieval operation shall block.
    // * `future_state_value` indicates a value has been set and can be read.
    //   Any retrieval operation shall unblock and return the value.
    // * `future_state_except` indicates either an exception has been set or the
    //   associated promise went out of scope without setting a value.
    //   Any retrieval operation shall unblock and throw an exception.
    ROCKET_PURE_FUNCTION
    Future_State
    state()
    const noexcept override
      {
        mutex::unique_lock lock(this->m_mutex);
        return static_cast<Future_State>(this->m_stor.index());
      }

    // Retrieves the value, if one has been set.
    // If no value has been set, an exception is thrown, and there is no effect.
    typename ::std::add_lvalue_reference<const ValueT>::type
    get_value()
    const
      {
        mutex::unique_lock lock(this->m_mutex);
        if(this->m_stor.index() != future_state_value)
          this->do_throw_valueless_unlocked();

        // The cast is necessary when `ValueT` is `void`.
        return static_cast<typename ::std::add_lvalue_reference<const ValueT>::type>(
                     this->m_stor.template as<future_state_value>());
      }

    ValueT
    copy_value()
    const
      {
        mutex::unique_lock lock(this->m_mutex);
        if(this->m_stor.index() != future_state_value)
          this->do_throw_valueless_unlocked();

        // The cast is necessary when `ValueT` is `void`.
        return static_cast<typename ::std::add_lvalue_reference<const ValueT>::type>(
                     this->m_stor.template as<future_state_value>());
      }

    typename ::std::add_lvalue_reference<ValueT>::type
    open_value()
      {
        mutex::unique_lock lock(this->m_mutex);
        if(this->m_stor.index() != future_state_value)
          this->do_throw_valueless_unlocked();

        // The cast is necessary when `ValueT` is `void`.
        return static_cast<typename ::std::add_lvalue_reference<ValueT>::type>(
                     this->m_stor.template as<future_state_value>());
      }

    ValueT
    move_value()
      {
        mutex::unique_lock lock(this->m_mutex);
        if(this->m_stor.index() != future_state_value)
          this->do_throw_valueless_unlocked();

        // The cast is necessary when `ValueT` is `void`.
        return static_cast<typename ::std::add_rvalue_reference<ValueT>::type>(
                     this->m_stor.template as<future_state_value>());
      }
  };

template<typename ValueT>
Future<ValueT>::
~Future()
  = default;

}  // namespace poseidon

#endif
