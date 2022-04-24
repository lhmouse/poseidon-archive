// This file is part of Poseidon.
// Copyleft 2020, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_CORE_LCG48_HPP_
#define POSEIDON_CORE_LCG48_HPP_

#include "../fwd.hpp"

namespace poseidon {

class LCG48
  {
  public:
    // This class is a UniformRandomBitGenerator.
    using result_type  = uint32_t;

    static constexpr
    result_type
    min() noexcept
      { return 0; }

    static constexpr
    result_type
    max() noexcept
      { return UINT32_MAX;  }

    static
    uint64_t
    create_seed() noexcept;

  private:
    uint64_t m_seed;

  public:
    explicit constexpr
    LCG48(uint64_t seed = create_seed()) noexcept
      : m_seed(seed)
      { }

  public:
    // These functions can be used to save and restore states.
    constexpr
    uint64_t
    get_seed() const noexcept
      { return this->m_seed;  }

    LCG48&
    set_seed(uint64_t seed) noexcept
      { return this->m_seed = seed, *this;  }

    // Gets a pseudo random number.
    uint32_t
    bump() noexcept;

    result_type
    operator()() noexcept
      { return this->bump();  }

    LCG48&
    swap(LCG48& other) noexcept
      {
        ::std::swap(this->m_seed, other.m_seed);
        return *this;
      }
  };

inline
void
swap(LCG48& lhs, LCG48& rhs) noexcept
  { lhs.swap(rhs);  }

}  // namespace poseidon

#endif
