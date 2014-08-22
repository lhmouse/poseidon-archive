#include "../precompiled.hpp"
#include "utilities.hpp"
#include "log.hpp"
#include <time.h>
#include <pthread.h>
#include <errno.h>
#include <string.h>
using namespace Poseidon;

namespace Poseidon {

namespace {

struct TzSetHelper {
	TzSetHelper(){
		::tzset();
	}
} const g_tzSetHelper;

}

boost::uint64_t getUtcTime(){
	::timespec ts;
	if(::clock_gettime(CLOCK_REALTIME, &ts) != 0){
		LOG_FATAL <<"Realtime clock is not supported.";
		std::abort();
	}
	return ts.tv_sec * 1000 + ts.tv_nsec / 1000000;
}
boost::uint64_t getLocalTime(){
	return getLocalTimeFromUtc(getUtcTime());
}
boost::uint64_t getUtcTimeFromLocal(boost::uint64_t local){
	return local + ::timezone * 1000;
}
boost::uint64_t getLocalTimeFromUtc(boost::uint64_t utc){
	return utc - ::timezone * 1000;
}

boost::uint64_t getMonoClock(){
	::timespec ts;
	if(::clock_gettime(CLOCK_MONOTONIC, &ts) != 0){
		LOG_FATAL <<"Monotonic clock is not supported.";
		std::abort();
	}
	return ts.tv_sec * 1000000 + ts.tv_nsec / 1000;
}

namespace {

class RandSeed : boost::noncopyable {
private:
	pthread_key_t m_key;

public:
	RandSeed(){
		const int code = ::pthread_key_create(&m_key, NULL);
		if(code != 0){
			char temp[256];
			const char *const desc = ::strerror_r(errno, temp, sizeof(temp));
			LOG_FATAL <<"Error allocating thread specific key for rand seed: " <<desc;
			std::abort();
		}
	}
	~RandSeed(){
		:: pthread_key_delete(m_key);
	}

public:
	boost::uint32_t get() const {
		return (std::size_t)::pthread_getspecific(m_key);
	}
	void set(boost::uint32_t val){
		::pthread_setspecific(m_key, (void *)(std::size_t)val);
	}
} g_randSeed;

}

boost::uint32_t rand32(){
	boost::uint64_t seed = g_randSeed.get();
	if(seed == 0){
		seed = getMonoClock();
	}
	// MMIX by Donald Knuth
	seed = seed * 6364136223846793005ull + 1442695040888963407ull;
	g_randSeed.set(seed);
	return seed >> 32;
}
boost::uint64_t rand64(){
	return (boost::uint64_t)rand32() | rand32();
}
boost::uint32_t rand32(boost::uint32_t lower, boost::uint32_t upper){
	if(lower > upper){
		boost::uint32_t tmp = lower;
		lower = upper + 1;
		upper = tmp - 1;
	}
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
	if(lower > upper){
		double tmp = lower;
		lower = upper;
		upper = tmp;
	}
	AUTO(const delta, upper - lower);
	if(delta == 0){
		return lower;
	}
	return lower + rand64() / 0x1p64 * delta;
}

}
