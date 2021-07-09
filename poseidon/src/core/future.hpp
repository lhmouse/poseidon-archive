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
    explicit
    Future() noexcept
      = default;

  public:
    ASTERIA_NONCOPYABLE_DESTRUCTOR(Future);

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
    ROCKET_PURE Future_State
    state() const noexcept final
      {
        simple_mutex::unique_lock lock(this->m_mutex);
        return static_cast<Future_State>(this->m_stor.index());
      }

    // Retrieves the value, if one has been set.
    // If no value has been set, an exception is thrown, and there is no effect.
    typename ::std::add_lvalue_reference<const ValueT>::type
    value() const
      {
        simple_mutex::unique_lock lock(this->m_mutex);
        switch(this->m_stor.index()) {
          case future_state_empty:
            // Nothing has been set yet.
            ::rocket::sprintf_and_throw<::std::invalid_argument>(
                  "Future: No value set yet (value type `%s`)",
                  typeid(ValueT).name());

          case future_state_value:
            // The cast is necessary when `ValueT` is `void`.
            return static_cast<typename ::std::add_lvalue_reference<const ValueT>::type>(
                         this->m_stor.template as<future_state_value>());

          case future_state_except: {
            // This state indicates either an exception has been set or the associated
            // promise went out of scope without setting a value.
            const auto& eptr = this->m_stor.template as<future_state_except>();
            if(eptr)
              ::std::rethrow_exception(eptr);

            // Report broken promise if a null exception pointer has been set,
            // anticipatedly by the destructor of the associated promise.
            ::rocket::sprintf_and_throw<::std::invalid_argument>(
                  "Future: Broken promise (value type `%s`)",
                  typeid(ValueT).name());
          }

          default:
            ROCKET_ASSERT(false);
        }
        ROCKET_UNREACHABLE();
      }
  };

template<typename ValueT>
Future<ValueT>::
~Future()
  = default;

}  // namespace poseidon

#endif
