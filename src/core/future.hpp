// This file is part of Poseidon.
// Copyleft 2020, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_CORE_FUTURE_
#define POSEIDON_CORE_FUTURE_

#include "abstract_future.hpp"

namespace poseidon {

template<typename ValueT>
class Future
  : public ::asteria::Rcfwd<Future<ValueT>>,
    public Abstract_Future
  {
    friend Promise<ValueT>;

  private:
    union {
      typename ::std::conditional<::std::is_void<ValueT>::value, int, ValueT>::type m_value;
      ::std::exception_ptr m_exptr;
    };

  public:
    explicit
    Future() noexcept
      {
        ROCKET_ASSERT(this->m_state.load() == future_state_empty);
      }

    Future(const Future&)
      = delete;

    Future&
    operator=(const Future&)
      = delete;

  public:
    ~Future();

    // Retrieves the value, if one has been set.
    // If no value has been set, an exception is thrown, and there is no effect.
    typename ::std::add_lvalue_reference<const ValueT>::type
    value() const
      {
        switch(this->m_state.load()) {
          case future_state_empty:
            // Nothing has been set yet.
            ::rocket::sprintf_and_throw<::std::invalid_argument>(
                  "Future: no value set yet (value type `%s`)",
                  typeid(ValueT).name());

          case future_state_value:
            // The cast is necessary when `ValueT` is `void`.
            return (typename ::std::add_lvalue_reference<const ValueT>::type) this->m_value;

          case future_state_except:
            // This state indicates either an exception has been set or the associated
            // promise went out of scope without setting a value.
            if(this->m_exptr)
              ::std::rethrow_exception(this->m_exptr);
            else
              ::rocket::sprintf_and_throw<::std::invalid_argument>(
                    "Future: broken promise (value type `%s`)",
                    typeid(ValueT).name());
        }
        ROCKET_UNREACHABLE();
      }

    typename ::std::add_lvalue_reference<ValueT>::type
    value()
      {
        switch(this->m_state.load()) {
          case future_state_empty:
            // Nothing has been set yet.
            ::rocket::sprintf_and_throw<::std::invalid_argument>(
                  "Future: no value set yet (value type `%s`)",
                  typeid(ValueT).name());

          case future_state_value:
            // The cast is necessary when `ValueT` is `void`.
            return (typename ::std::add_lvalue_reference<ValueT>::type) this->m_value;

          case future_state_except:
            // This state indicates either an exception has been set or the associated
            // promise went out of scope without setting a value.
            if(this->m_exptr)
              ::std::rethrow_exception(this->m_exptr);
            else
              ::rocket::sprintf_and_throw<::std::invalid_argument>(
                    "Future: broken promise (value type `%s`)",
                    typeid(ValueT).name());
        }
        ROCKET_UNREACHABLE();
      }
  };

template<typename ValueT>
Future<ValueT>::
~Future()
  {
    switch(this->m_state.load()) {
      case future_state_empty:
        return;

      case future_state_value:
        ::rocket::destroy(::std::addressof(this->m_value));
        return;

      case future_state_except:
        ::rocket::destroy(&(this->m_exptr));
        return;
    }
  }

}  // namespace poseidon

#endif
