// This file is part of Poseidon.
// Copyleft 2020, LH_Mouse. All wrongs reserved.

#include "../precompiled.hpp"
#include "lcg48.hpp"
#include "../util.hpp"

namespace poseidon {

uint64_t
LCG48::
create_seed()
  noexcept
  {
    // Use the number of nanoseconds as the seed.
    ::timespec ts = { };
    ::clock_gettime(CLOCK_MONOTONIC, &ts);
    return static_cast<uint64_t>(ts.tv_nsec);
  }

uint32_t
LCG48::
bump()
  noexcept
  {
    // These arguments are the same as glibc's `drand48()` function.
    uint64_t seed = this->m_seed * 0x5DEECE66D + 0xB;
    this->m_seed = seed & 0xFFFFFFFFFFFF;
    return static_cast<uint32_t>(seed >> 16);
  }

}  // namespace poseidon
