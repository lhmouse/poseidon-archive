// This file is part of Poseidon.
// Copyleft 2020, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_CORE_FUTURE_
#define POSEIDON_CORE_FUTURE_

#include "../fwd.hpp"

namespace poseidon {

template<typename ValueT>
class Future
  : public Abstract_Future
  {
  private:
    union {
      // This member is active if `future_state() == future_state_value`.
      typename ::std::conditional<
          ::std::is_void<ValueT>::value, int, ValueT>::type m_value[1];

      // This member is active if `future_state() == future_state_exception`.
      exception_ptr m_exception[1];
    };

  public:
    // Constructs an empty future.
    Future() noexcept;

  private:
    int
    do_check_throw_common() const
      {
        switch(this->m_future_state.load()) {
          case future_state_empty:
            // Fail.
            ::rocket::sprintf_and_throw<::std::invalid_argument>(
                "Future: No value set\n"
                "[value type `%s`]",
                typeid(ValueT).name());

          case future_state_value:
            return 0;

          case future_state_exception:
            // Throw the caught exception.
            if(this->m_exception[0])
              ::std::rethrow_exception(this->m_exception[0]);
            else
              ::rocket::sprintf_and_throw<::std::invalid_argument>(
                  "Future: Promise broken without an exception\n"
                  "[value type `%s`]",
                  typeid(ValueT).name());

          default:
            ::std::terminate();
        }
      }

  public:
    ASTERIA_NONCOPYABLE_VIRTUAL_DESTRUCTOR(Future);

    // Gets the value if one has been set, or throws an exception otherwise.
    typename ::std::add_lvalue_reference<const ValueT>::type
    value() const
      {
        this->do_check_throw_common();

        // The cast is necessary when `ValueT` is void.
        return static_cast<typename
           ::std::add_lvalue_reference<const ValueT>::type>(this->m_value[0]);
      }

    // Gets the value if one has been set, or throws an exception otherwise.
    typename ::std::add_lvalue_reference<ValueT>::type
    value()
      {
        this->do_check_throw_common();

        // The cast is necessary when `ValueT` is void.
        return static_cast<typename
           ::std::add_lvalue_reference<ValueT>::type>(this->m_value[0]);
      }

    // Sets a value.
    // If a value or exception has already been set, this function does nothing.
    template<typename... ParamsT>
    void
    set_value(ParamsT&&... params)
      {
        this->m_once.call(
          [&] {
            // Construct the value.
            ROCKET_ASSERT(this->m_future_state.load() == future_state_empty);
            ::rocket::construct(this->m_value, ::std::forward<ParamsT>(params)...);
            this->m_future_state.store(future_state_value);
          });
      }

    // Sets an exception.
    // If a value or exception has already been set, this function does nothing.
    void
    set_exception(const ::std::exception_ptr& exception_opt) noexcept
      {
        this->m_once.call(
          [&] {
            // Construct the exception.
            ROCKET_ASSERT(this->m_future_state.load() == future_state_empty);
            ::rocket::construct(this->m_exception, exception_opt);
            this->m_future_state.store(future_state_exception);
          });
      }
  };

template<typename ValueT>
Future<ValueT>::
Future() noexcept
  {
  }

template<typename ValueT>
Future<ValueT>::
~Future()
  {
    switch(this->m_future_state.load()) {
      case future_state_empty:
        return;

      case future_state_value:
        // Destroy the value that has been constructed.
        ::rocket::destroy(this->m_value);
        return;

      case future_state_exception:
        // Destroy the exception that has been constructed.
        ::rocket::destroy(this->m_exception);
        return;

      default:
        ::std::terminate();
    }
  }

}  // namespace poseidon

#endif
