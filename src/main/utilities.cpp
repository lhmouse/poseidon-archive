#include "../precompiled.hpp"
#include "utilities.hpp"
#include <boost/chrono/chrono.hpp>
#include <boost/thread/tss.hpp>
using namespace Poseidon;

namespace Poseidon {

boost::uint64_t getLocalTime(){
	return boost::chrono::nanoseconds(
		boost::chrono::system_clock::now().time_since_epoch()
	).count() / 1000000;
}
boost::uint64_t getMonoClock(){
	return boost::chrono::nanoseconds(
		boost::chrono::steady_clock::now().time_since_epoch()
	).count() / 1000;
}

namespace {
	boost::thread_specific_ptr<int> g_seed(NULL);
}

boost::uint32_t rand32(){
	boost::uint64_t seed = (std::size_t)g_seed.get();
	if(seed == 0){
		seed = getMonoClock();
	}
	// MMIX by Donald Knuth
	seed = seed * 6364136223846793005ull + 1442695040888963407ull;
	g_seed.reset((int *)(std::size_t)seed);
	return seed >> 32;
}
boost::uint64_t rand64(){
	return (boost::uint64_t)rand32() | rand32();
}
boost::uint32_t rand32(boost::uint32_t lower, boost::uint32_t upper){
	assert(lower <= upper);

	AUTO(const delta, upper - lower);
	if(delta == 0){
		return lower;
	}
	if(delta < 0x10000){
		return lower + rand32() % delta;
	}
	return lower + rand64() % delta;
}
double randDouble(double lower, double upper){
	assert(lower <= upper);

	AUTO(const delta, upper - lower);
	if(delta == 0){
		return lower;
	}
	return lower + rand64() / 0x1p64 * delta;
}

}
