// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2016, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_RANDOM_HPP_
#define POSEIDON_RANDOM_HPP_

#include <boost/cstdint.hpp>
#include "cxx_ver.hpp"

namespace Poseidon {

extern boost::uint32_t random_uint32();
extern boost::uint64_t random_uint64();
extern double random_double();

struct RandomBitGeneratorUint32 {
	typedef boost::uint32_t result_type;

	static CONSTEXPR result_type min() NOEXCEPT {
		return 0;
	}
	static CONSTEXPR result_type max() NOEXCEPT {
		return (result_type)-1;
	}

	result_type operator()() const NOEXCEPT {
		return random_uint32();
	}
};

struct RandomBitGeneratorUint64 {
	typedef boost::uint64_t result_type;

	static CONSTEXPR result_type min() NOEXCEPT {
		return 0;
	}
	static CONSTEXPR result_type max() NOEXCEPT {
		return (result_type)-1;
	}

	result_type operator()() const NOEXCEPT {
		return random_uint64();
	}
};

struct RandomBitGeneratorDouble {
	typedef double result_type;

	static CONSTEXPR result_type min() NOEXCEPT {
		return 0;
	}
	static CONSTEXPR result_type max() NOEXCEPT {
		return 1; // FIXME 应该是 1 - 0x1p-63，但是这个值不能用 double 表示。
	}

	result_type operator()() const NOEXCEPT {
		return random_double();
	}
};

}

#endif
