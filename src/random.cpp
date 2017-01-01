// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2017, LH_Mouse. All wrongs reserved.

#include "precompiled.hpp"
#include "random.hpp"

namespace Poseidon {

namespace {
	__thread boost::uint64_t t_random_seed = 0;
}

boost::uint32_t random_uint32(){
	boost::uint64_t seed = t_random_seed, tsc;
	__asm__ __volatile__(
		"rdtsc \n"
#ifdef __x86_64__
		"shlq $32, %%rdx \n"
		"orq %%rdx, %%rax \n"
		: "=a"(tsc) : : "dx"
#else
		: "=A"(tsc) : :
#endif
	);
	seed ^= tsc;
	// MMIX by Donald Knuth
	seed = seed * 6364136223846793005ull + 1442695040888963407ull;
	t_random_seed = seed;
	return seed >> 32;
}
boost::uint64_t random_uint64(){
	return (static_cast<boost::uint64_t>(random_uint32()) << 32) | random_uint32();
}
double random_double(){
	return static_cast<boost::int64_t>(random_uint64() >> 11) / 0x1p53;
}

}
