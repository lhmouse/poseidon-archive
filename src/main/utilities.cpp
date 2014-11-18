// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014, LH_Mouse. All wrongs reserved.

#include "precompiled.hpp"
#include "utilities.hpp"
#include "log.hpp"
#include <boost/thread/once.hpp>
#include <time.h>
#include <errno.h>
#include <string.h>
using namespace Poseidon;

namespace {

boost::once_flag g_tzsetFlag;

}

namespace Poseidon {

boost::uint64_t getUtcTime(){
	::timespec ts;
	if(::clock_gettime(CLOCK_REALTIME, &ts) != 0){
		LOG_POSEIDON_FATAL("Realtime clock is not supported.");
		std::abort();
	}
	return (boost::uint64_t)ts.tv_sec * 1000 + ts.tv_nsec / 1000000;
}
boost::uint64_t getLocalTime(){
	return getLocalTimeFromUtc(getUtcTime());
}
boost::uint64_t getUtcTimeFromLocal(boost::uint64_t local){
	boost::call_once(&::tzset, g_tzsetFlag);
	return local + ::timezone * 1000;
}
boost::uint64_t getLocalTimeFromUtc(boost::uint64_t utc){
	boost::call_once(&::tzset, g_tzsetFlag);
	return utc - ::timezone * 1000;
}

boost::uint64_t getMonoClock() NOEXCEPT {
	::timespec ts;
	if(::clock_gettime(CLOCK_MONOTONIC, &ts) != 0){
		LOG_POSEIDON_FATAL("Monotonic clock is not supported.");
		std::abort();
	}
	return (boost::uint64_t)ts.tv_sec * 1000000 + ts.tv_nsec / 1000;
}

namespace {

__thread boost::uint64_t t_randSeed = 0;

}

boost::uint32_t rand32(){
	boost::uint64_t seed = t_randSeed;
	if(seed == 0){
		seed = getMonoClock();
	}
	// MMIX by Donald Knuth
	seed = seed * 6364136223846793005ull + 1442695040888963407ull;
	t_randSeed = seed;
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

SharedNtmbs getErrorDesc(int errCode) NOEXCEPT {
	char temp[1024];
	const char *desc = ::strerror_r(errCode, temp, sizeof(temp));
	if(desc == temp){
		try {
			return SharedNtmbs(desc, true);
		} catch(...){
			desc = "Insufficient memory.";
		}
	}
	// desc 指向一个静态的字符串。
	return SharedNtmbs(desc);
}
std::string getErrorDescAsString(int errCode){
	std::string ret;
	ret.resize(1024);
	const char *desc = ::strerror_r(errCode, &ret[0], ret.size());
	if(desc == &ret[0]){
		ret.resize(std::strlen(desc));
	} else {
		ret.assign(desc);
	}
	return ret;
}

}
