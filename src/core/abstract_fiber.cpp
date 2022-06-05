// This file is part of Poseidon.
// Copyleft 2020, LH_Mouse. All wrongs reserved.

#include "../precompiled.ipp"
#include "abstract_fiber.hpp"
#include "../utils.hpp"

namespace poseidon {

Abstract_Fiber::
~Abstract_Fiber()
  {
  }

void
Abstract_Fiber::
do_abstract_fiber_on_start() noexcept
  {
  }

void
Abstract_Fiber::
do_abstract_fiber_on_suspend() noexcept
  {
  }

void
Abstract_Fiber::
do_abstract_fiber_on_resume() noexcept
  {
  }

void
Abstract_Fiber::
do_abstract_fiber_on_finish() noexcept
  {
  }

}  // namespace poseidon
