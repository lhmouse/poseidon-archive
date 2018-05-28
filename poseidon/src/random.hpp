// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2018, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_RANDOM_HPP_
#define POSEIDON_RANDOM_HPP_

#include <boost/cstdint.hpp>
#include "cxx_ver.hpp"

namespace Poseidon {

extern std::uint32_t random_uint32();
extern std::uint64_t random_uint64();
extern double random_double();

struct Random_bit_generator_uint32 {
	typedef std::uint32_t result_type;

	static CONSTEXPR result_type min() NOEXCEPT {
		return 0;
	}
	static CONSTEXPR result_type max() NOEXCEPT {
		return static_cast<result_type>(-1);
	}

	result_type operator()() const NOEXCEPT {
		return random_uint32();
	}
};

struct Random_bit_generator_uint64 {
	typedef std::uint64_t result_type;

	static CONSTEXPR result_type min() NOEXCEPT {
		return 0;
	}
	static CONSTEXPR result_type max() NOEXCEPT {
		return static_cast<result_type>(-1);
	}

	result_type operator()() const NOEXCEPT {
		return random_uint64();
	}
};

}

#endif
