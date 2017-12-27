// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2017, LH_Mouse. All wrongs reserved.

#include "precompiled.hpp"
#include "random.hpp"
#include "atomic.hpp"

namespace Poseidon {

namespace {
	volatile boost::uint64_t s_seed;
}

boost::uint32_t random_uint32(){
	boost::uint64_t old_seed, new_seed;
	{
		old_seed = atomic_load(s_seed, ATOMIC_RELAXED);
		do {
			new_seed = old_seed;
			new_seed ^= __builtin_ia32_rdtsc();
			// MMIX by Donald Knuth
			new_seed *= 6364136223846793005u;
			new_seed += 1442695040888963407u;
		} while(!atomic_compare_exchange(s_seed, old_seed, new_seed, ATOMIC_RELAXED, ATOMIC_RELAXED));
	}
	return static_cast<boost::uint32_t>(new_seed >> 32);
}
boost::uint64_t random_uint64(){
	return (static_cast<boost::uint64_t>(random_uint32()) << 32) + random_uint32();
}
double random_double(){
	return static_cast<double>(static_cast<boost::int64_t>(random_uint64() >> 1)) / 0x1p63;
}

}
