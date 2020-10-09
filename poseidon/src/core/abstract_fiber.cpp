// This file is part of Poseidon.
// Copyleft 2020, LH_Mouse. All wrongs reserved.

#include "../precompiled.hpp"
#include "abstract_fiber.hpp"
#include "../util.hpp"

namespace poseidon {

Abstract_Fiber::
~Abstract_Fiber()
  {
  }

void
Abstract_Fiber::
do_on_start()
noexcept
  {
  }

void
Abstract_Fiber::
do_on_suspend()
noexcept
  {
  }

void
Abstract_Fiber::
do_on_resume()
noexcept
  {
  }

void
Abstract_Fiber::
do_on_finish()
noexcept
  {
  }

}  // namespace poseidon
