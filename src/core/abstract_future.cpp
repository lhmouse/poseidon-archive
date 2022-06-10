// This file is part of Poseidon.
// Copyleft 2022, LH_Mouse. All wrongs reserved.

#include "../precompiled.ipp"
#include "abstract_future.hpp"
#include "../utils.hpp"

namespace poseidon {

Abstract_Future::
Abstract_Future() noexcept
  {
  }

Abstract_Future::
~Abstract_Future()
  {
  }

void
Abstract_Future::
do_throw_future_exception(const type_info& type, const exception_ptr* exptr) const
  {
    switch(this->m_future_state.load()) {
      case future_state_empty:
        POSEIDON_THROW(
            "No value has been set\n"
            "[value type `$1`]",
            type);

      case future_state_value:
        ROCKET_ASSERT(false);

      case future_state_exception:
        // `*exptr` is valid.
        if(*exptr)
          ::std::rethrow_exception(*exptr);
        else
          POSEIDON_THROW(
              "Promise broken without an exception\n"
              "[value type `$1`]",
              type);

      default:
        ROCKET_ASSERT(false);
    }
  }

}  // namespace poseidon
