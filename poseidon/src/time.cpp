// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2018, LH_Mouse. All wrongs reserved.

#include "precompiled.hpp"
#include "time.hpp"
#include "log.hpp"
#include "exception.hpp"
#include <stdio.h>

namespace Poseidon {

namespace {
	::pthread_once_t g_tz_once = PTHREAD_ONCE_INIT;
}

std::uint64_t get_utc_time(){
	::timespec ts;
	if(::clock_gettime(CLOCK_REALTIME, &ts) != 0){
		POSEIDON_LOG_FATAL("Realtime clock is not supported.");
		std::terminate();
	}
	return (std::uint64_t)(ts.tv_sec * 1000 + ts.tv_nsec / 1000000);
}
std::uint64_t get_local_time(){
	return get_local_time_from_utc(get_utc_time());
}
std::uint64_t get_utc_time_from_local(std::uint64_t local){
	if(::pthread_once(&g_tz_once, &::tzset) != 0){
		POSEIDON_LOG_FATAL("::pthread_once() failed.");
		std::terminate();
	}
	return local + (std::uint64_t)(::timezone * 1000);
}
std::uint64_t get_local_time_from_utc(std::uint64_t utc){
	if(::pthread_once(&g_tz_once, &::tzset) != 0){
		POSEIDON_LOG_FATAL("::pthread_once() failed.");
		std::terminate();
	}
	return utc - (std::uint64_t)(::timezone * 1000);
}

// 这里沿用了 MCF 的旧称。在 Windows 上 get_fast_mono_clock() 是 GetTickCount64() 实现的。
std::uint64_t get_fast_mono_clock() NOEXCEPT {
	::timespec ts;
	if(::clock_gettime(CLOCK_MONOTONIC, &ts) != 0){
		POSEIDON_LOG_FATAL("Monotonic clock is not supported.");
		std::terminate();
	}
	return (std::uint64_t)(ts.tv_sec * 1000 + ts.tv_nsec / 1000000);
}
// 在 Windows 上 get_hi_res_mono_clock() 是 Query_performance_counter() 实现的。
double get_hi_res_mono_clock() NOEXCEPT {
	::timespec ts;
	if(::clock_gettime(CLOCK_MONOTONIC, &ts) != 0){
		POSEIDON_LOG_FATAL("Monotonic clock is not supported.");
		std::terminate();
	}
	return (double)ts.tv_sec * 1e3 + (double)ts.tv_nsec / 1e6;
}

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wsign-conversion"

Date_time break_down_time(std::uint64_t ms){
	Date_time dt = { 1234, 1, 1, 0, 0, 0, 0 };
	if(ms == 0){
		dt.yr = 0;
	} else if(ms == -1ull){
		dt.yr = 9999;
	} else {
		const AUTO(dt_sec, (::time_t)(ms / 1000));
		const AUTO(dt_ms, (unsigned)(ms % 1000));
		::tm desc;
		::gmtime_r(&dt_sec, &desc);
		dt.yr  = 1900 + desc.tm_year;
		dt.mon = 1 + desc.tm_mon;
		dt.day = desc.tm_mday;
		dt.hr  = desc.tm_hour;
		dt.min = desc.tm_min;
		dt.sec = desc.tm_sec;
		dt.ms  = dt_ms;
	}
	return dt;
}
std::uint64_t assemble_time(const Date_time &dt){
	std::uint64_t ms;
	if(dt.yr == 0){
		ms = 0;
	} else if(dt.yr == 9999){
		ms = -1ull;
	} else {
		::tm desc;
		desc.tm_year = dt.yr - 1900;
		desc.tm_mon  = dt.mon - 1;
		desc.tm_mday = dt.day;
		desc.tm_hour = dt.hr;
		desc.tm_min  = dt.min;
		desc.tm_sec  = dt.sec;
		ms = (std::uint64_t)::timegm(&desc) * 1000 + dt.ms;
	}
	return ms;
}

#pragma GCC diagnostic pop

std::size_t format_time(char *buffer, std::size_t max, std::uint64_t ms, bool show_ms){
	Date_time dt = break_down_time(ms);
	return (unsigned)std::snprintf(buffer, max, show_ms ? "%04u-%02u-%02u %02u:%02u:%02u.%03u"
	                                                 : "%04u-%02u-%02u %02u:%02u:%02u",
		dt.yr, dt.mon, dt.day, dt.hr, dt.min, dt.sec, dt.ms);
}
std::uint64_t scan_time(const char *str){
	Date_time dt = { 1234, 1, 1, 0, 0, 0, 0 };
	if(std::sscanf(str, "%u-%u-%u %u:%u:%u.%u", &dt.yr, &dt.mon, &dt.day, &dt.hr, &dt.min, &dt.sec, &dt.ms) < 3){
		POSEIDON_LOG_ERROR("Time string is not valid: ", str);
		POSEIDON_THROW(Exception, Rcnts::view("Time string is not valid"));
	}
	return assemble_time(dt);
}

}
