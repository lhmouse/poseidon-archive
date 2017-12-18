// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2017, LH_Mouse. All wrongs reserved.

#include "precompiled.hpp"
#include "time.hpp"
#include "log.hpp"
#include <stdio.h>

namespace Poseidon {

namespace {
	::pthread_once_t g_tz_once = PTHREAD_ONCE_INIT;
}

boost::uint64_t get_utc_time(){
	::timespec ts;
	if(::clock_gettime(CLOCK_REALTIME, &ts) != 0){
		LOG_POSEIDON_FATAL("Realtime clock is not supported.");
		std::abort();
	}
	return (boost::uint64_t)(ts.tv_sec * 1000 + ts.tv_nsec / 1000000);
}
boost::uint64_t get_local_time(){
	return get_local_time_from_utc(get_utc_time());
}
boost::uint64_t get_utc_time_from_local(boost::uint64_t local){
	const int err = ::pthread_once(&g_tz_once, &::tzset);
	if(err != 0){
		LOG_POSEIDON_FATAL("::pthread_once() failed with error code ", err);
		std::abort();
	}
	return local + (boost::uint64_t)(::timezone * 1000);
}
boost::uint64_t get_local_time_from_utc(boost::uint64_t utc){
	const int err = ::pthread_once(&g_tz_once, &::tzset);
	if(err != 0){
		LOG_POSEIDON_FATAL("::pthread_once() failed with error code ", err);
		std::abort();
	}
	return utc - (boost::uint64_t)(::timezone * 1000);
}

// 这里沿用了 MCF 的旧称。在 Windows 上 get_fast_mono_clock() 是 GetTickCount64() 实现的。
boost::uint64_t get_fast_mono_clock() NOEXCEPT {
	::timespec ts;
	if(::clock_gettime(CLOCK_MONOTONIC, &ts) != 0){
		LOG_POSEIDON_FATAL("Monotonic clock is not supported.");
		std::abort();
	}
	return (boost::uint64_t)(ts.tv_sec * 1000 + ts.tv_nsec / 1000000);
}
// 在 Windows 上 get_hi_res_mono_clock() 是 QueryPerformanceCounter() 实现的。
double get_hi_res_mono_clock() NOEXCEPT {
	::timespec ts;
	if(::clock_gettime(CLOCK_MONOTONIC, &ts) != 0){
		LOG_POSEIDON_FATAL("Monotonic clock is not supported.");
		std::abort();
	}
	return (double)ts.tv_sec * 1e3 + (double)ts.tv_nsec / 1e6;
}

DateTime break_down_time(boost::uint64_t ms){
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
boost::uint64_t assemble_time(const DateTime &dt){
	::tm desc;
	desc.tm_year = static_cast<int>(dt.yr - 1900);
	desc.tm_mon  = static_cast<int>(dt.mon - 1);
	desc.tm_mday = static_cast<int>(dt.day);
	desc.tm_hour = static_cast<int>(dt.hr);
	desc.tm_min  = static_cast<int>(dt.min);
	desc.tm_sec  = static_cast<int>(dt.sec);
	return static_cast<boost::uint64_t>(::timegm(&desc)) * 1000 + dt.ms;
}

std::size_t format_time(char *buffer, std::size_t max, boost::uint64_t ms, bool show_ms){
	DateTime dt = { 1234, 1, 1, 0, 0, 0, 0 };
	if(ms == 0){
		dt.yr = 0;
	} else if(ms == (boost::uint64_t)-1){
		dt.yr = 9999;
	} else {
		dt = break_down_time(ms);
	}
	return (unsigned)::snprintf(buffer, max, show_ms ? "%04u-%02u-%02u %02u:%02u:%02u.%03u"
	                                                 : "%04u-%02u-%02u %02u:%02u:%02u",
		dt.yr, dt.mon, dt.day, dt.hr, dt.min, dt.sec, dt.ms);
}
boost::uint64_t scan_time(const char *str){
	DateTime dt = { 1234, 1, 1, 0, 0, 0, 0 };
	std::sscanf(str, "%u-%u-%u %u:%u:%u.%u", &dt.yr, &dt.mon, &dt.day, &dt.hr, &dt.min, &dt.sec, &dt.ms);
	if(dt.yr == 0){
		return 0;
	} else if(dt.yr == 9999){
		return (boost::uint64_t)-1;
	} else {
		return assemble_time(dt);
	}
}

}
