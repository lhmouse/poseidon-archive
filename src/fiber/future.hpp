// This file is part of Poseidon.
// Copyleft 2022, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_FIBER_FUTURE_
#define POSEIDON_FIBER_FUTURE_

#include "../fwd.hpp"
#include "abstract_future.hpp"

namespace poseidon {

template<typename ValueT>
class future
  : public Abstract_Future
  {
    using value_type       = ValueT;
    using const_reference  = typename ::std::add_lvalue_reference<const ValueT>::type;
    using reference        = typename ::std::add_lvalue_reference<ValueT>::type;

  private:
    union {
      // This member is active if `future_state() == future_state_value`.
      typename ::std::conditional<
          ::std::is_void<ValueT>::value, int, ValueT>::type m_value[1];

      // This member is active if `future_state() == future_state_exception`.
      exception_ptr m_exptr[1];
    };

  public:
    // Constructs an empty future.
    explicit
    future() noexcept;

  public:
    ASTERIA_NONCOPYABLE_VIRTUAL_DESTRUCTOR(future);

    // Gets the value if one has been set, or throws an exception otherwise.
    const_reference
    value() const
      {
        // If no value has been set, throw an exception.
        if(this->m_state.load() != future_state_value)
          this->do_abstract_future_check_value(typeid(ValueT).name(), this->m_exptr);

        // This cast is necessary when `const_reference` is void.
        return (const_reference) this->m_value[0];
      }

    // Gets the value if one has been set, or throws an exception otherwise.
    reference
    value()
      {
        // If no value has been set, throw an exception.
        if(this->m_state.load() != future_state_value)
          this->do_abstract_future_check_value(typeid(ValueT).name(), this->m_exptr);

        // This cast is necessary when `reference` is void.
        return (reference) this->m_value[0];
      }

    // Sets a value.
    template<typename... ParamsT>
    void
    set_value(ParamsT&&... params)
      {
        // If a value or exception has already been set, this function shall
        // do nothing.
        plain_mutex::unique_lock lock(this->m_mutex);
        if(this->m_state.load() != future_state_empty)
          return;

        // Construct the value.
        ::rocket::construct(this->m_value, ::std::forward<ParamsT>(params)...);
        this->m_state.store(future_state_value);
        this->do_abstract_future_signal_nolock();
      }

    // Sets an exception.
    void
    set_exception(const ::std::exception_ptr& exptr_opt) noexcept
      {
        // If a value or exception has already been set, this function shall
        // do nothing.
        plain_mutex::unique_lock lock(this->m_mutex);
        if(this->m_state.load() != future_state_empty)
          return;

        // Construct the exception pointer.
        ::rocket::construct(this->m_exptr, exptr_opt);
        this->m_state.store(future_state_exception);
        this->do_abstract_future_signal_nolock();
      }
  };

template<typename ValueT>
future<ValueT>::
future() noexcept
  {
  }

template<typename ValueT>
future<ValueT>::
~future()
  {
    switch(this->m_state.load()) {
      case future_state_empty:
        break;

      case future_state_value:
        // Destroy the value that has been constructed.
        ::rocket::destroy(this->m_value);
        break;

      case future_state_exception:
        // Destroy the exception that has been constructed.
        ::rocket::destroy(this->m_exptr);
        break;
    }
  }

}  // namespace poseidon

#endif
