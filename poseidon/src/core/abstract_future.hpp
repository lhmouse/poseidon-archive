// This file is part of Poseidon.
// Copyleft 2020, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_CORE_ABSTRACT_FUTURE_HPP_
#define POSEIDON_CORE_ABSTRACT_FUTURE_HPP_

#include "../fwd.hpp"

namespace poseidon {

class Abstract_Future
  : public ::asteria::Rcfwd<Abstract_Future>
  {
  protected:
    mutable Si_Mutex m_mutex;

  public:
    Abstract_Future()
    noexcept
      { }

    ASTERIA_NONCOPYABLE_DESTRUCTOR(Abstract_Future);

  public:
    // Gets the state, which is any of `future_state_empty`, `future_state_value`
    // or `future_state_except`.
    ROCKET_PURE_FUNCTION virtual
    Future_State
    state()
    const noexcept
      = 0;
  };

}  // namespace poseidon

#endif
