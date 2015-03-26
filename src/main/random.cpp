// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2015, LH_Mouse. All wrongs reserved.

#include "precompiled.hpp"
#include "random.hpp"

namespace Poseidon {

namespace {
	__thread boost::uint64_t t_randSeed = 0;
}

boost::uint32_t rand32(){
	boost::uint64_t seed = t_randSeed;
	if(seed == 0){
		__asm__ __volatile__(
			"rdtsc \n"
#ifdef __x86_64__
			"shlq $32, %%rdx \n"
			"orq %%rdx, %%rax \n"
			: "=a"(seed) : : "dx"
#else
			: "=A"(seed) : :
#endif
		);
		seed |= 0x10001;
	}
	// MMIX by Donald Knuth
	seed = seed * 6364136223846793005ull + 1442695040888963407ull;
	t_randSeed = seed;
	return seed >> 32;
}
boost::uint64_t rand64(){
	return ((boost::uint64_t)rand32() << 32) | rand32();
}
boost::uint32_t rand32(boost::uint32_t lower, boost::uint32_t upper){
	if(lower > upper){
		boost::uint32_t tmp = lower;
		lower = upper + 1;
		upper = tmp - 1;
	}
	const AUTO(delta, upper - lower);
	if(delta == 0){
		return lower;
	}
	if(delta < 0x10000){
		return lower + rand32() % delta;
	}
	return lower + rand64() % delta;
}
double randDouble(double lower, double upper){
	if(lower > upper){
		double tmp = lower;
		lower = upper;
		upper = tmp;
	}
	const AUTO(delta, upper - lower);
	if(delta == 0){
		return lower;
	}
	return lower + rand64() / 0x1p64 * delta;
}

}
