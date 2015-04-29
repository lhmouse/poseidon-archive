// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2015, LH_Mouse. All wrongs reserved.

#include "precompiled.hpp"
#include "time.hpp"
#include "log.hpp"
#include <pthread.h>
#include <time.h>
#include <string.h>
#include <stdio.h>

namespace Poseidon {

namespace {
	::pthread_once_t g_tzOnce = PTHREAD_ONCE_INIT;
}

boost::uint64_t getUtcTime(){
	::timespec ts;
	if(::clock_gettime(CLOCK_REALTIME, &ts) != 0){
		LOG_POSEIDON_FATAL("Realtime clock is not supported.");
		std::abort();
	}
	return (boost::uint64_t)ts.tv_sec * 1000 + (unsigned long)ts.tv_nsec / 1000000;
}
boost::uint64_t getLocalTime(){
	return getLocalTimeFromUtc(getUtcTime());
}
boost::uint64_t getUtcTimeFromLocal(boost::uint64_t local){
	const int err = ::pthread_once(&g_tzOnce, &::tzset);
	if(err != 0){
		LOG_POSEIDON_FATAL("::pthread_once() failed with error code ", err);
		std::abort();
	}
	return local + (unsigned long)::timezone * 1000;
}
boost::uint64_t getLocalTimeFromUtc(boost::uint64_t utc){
	const int err = ::pthread_once(&g_tzOnce, &::tzset);
	if(err != 0){
		LOG_POSEIDON_FATAL("::pthread_once() failed with error code ", err);
		std::abort();
	}
	return utc - (unsigned long)::timezone * 1000;
}

// 这里沿用了 MCF 的旧称。在 Windows 上 getFastMonoClock() 是 GetTickCount64() 实现的。
boost::uint64_t getFastMonoClock() NOEXCEPT {
	::timespec ts;
	if(::clock_gettime(CLOCK_MONOTONIC, &ts) != 0){
		LOG_POSEIDON_FATAL("Monotonic clock is not supported.");
		std::abort();
	}
	return (boost::uint64_t)ts.tv_sec * 1000 + (unsigned long)ts.tv_nsec / 1000000;
}
// 在 Windows 上 getHiResMonoClock() 是 QueryPerformanceCounter() 实现的。
double getHiResMonoClock() NOEXCEPT {
	::timespec ts;
	if(::clock_gettime(CLOCK_MONOTONIC, &ts) != 0){
		LOG_POSEIDON_FATAL("Monotonic clock is not supported.");
		std::abort();
	}
	return (double)ts.tv_sec + (double)ts.tv_nsec / 1.0e9;
}

DateTime breakDownTime(boost::uint64_t ms){
	const ::time_t seconds = static_cast< ::time_t>(ms / 1000);
	const unsigned milliseconds = ms % 1000;
	::tm desc;
	::gmtime_r(&seconds, &desc);
	DateTime dt;
	dt.yr  = static_cast<unsigned>(1900 + desc.tm_year);
	dt.mon = static_cast<unsigned>(1 + desc.tm_mon);
	dt.day = static_cast<unsigned>(desc.tm_mday);
	dt.hr  = static_cast<unsigned>(desc.tm_hour);
	dt.min = static_cast<unsigned>(desc.tm_min);
	dt.sec = static_cast<unsigned>(desc.tm_sec);
	dt.ms  = static_cast<unsigned>(milliseconds);
	return dt;
}
boost::uint64_t assembleTime(const DateTime &dt){
	::tm desc;
	desc.tm_year = static_cast<int>(dt.yr - 1900);
	desc.tm_mon  = static_cast<int>(dt.mon - 1);
	desc.tm_mday = static_cast<int>(dt.day);
	desc.tm_hour = static_cast<int>(dt.hr);
	desc.tm_min  = static_cast<int>(dt.min);
	desc.tm_sec  = static_cast<int>(dt.sec);
	return static_cast<boost::uint64_t>(::timegm(&desc)) * 1000 + dt.ms;
}

std::size_t formatTime(char *buffer, std::size_t max, boost::uint64_t ms, bool showMs){
	DateTime dt;
	if(ms == 0){
		std::memset(&dt, 0, sizeof(dt));
	} else if(ms == (boost::uint64_t)-1){
		std::memset(&dt, 0, sizeof(dt));
		dt.yr = 9999;
	} else {
		dt = breakDownTime(ms);
	}
	return (std::size_t)::snprintf(buffer, max,
		showMs ? "%04u-%02u-%02u %02u:%02u:%02u.%03u" : "%04u-%02u-%02u %02u:%02u:%02u",
		dt.yr, dt.mon, dt.day, dt.hr, dt.min, dt.sec, dt.ms);
}
boost::uint64_t scanTime(const char *str){
	DateTime dt;
	std::memset(&dt, 0, sizeof(dt));
	std::sscanf(str, "%u-%u-%u %u:%u:%u.%u",
		&dt.yr, &dt.mon, &dt.day, &dt.hr, &dt.min, &dt.sec, &dt.ms);
	if(dt.yr == 0){
		return 0;
	} else if(dt.yr == 9999){
		return (boost::uint64_t)-1;
	} else {
		return assembleTime(dt);
	}
}

}
