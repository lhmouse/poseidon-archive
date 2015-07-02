// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2015, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_TIME_HPP_
#define POSEIDON_TIME_HPP_

#include "cxx_ver.hpp"
#include <boost/cstdint.hpp>

namespace Poseidon {

// 时间单位是毫秒。

boost::uint64_t getUtcTime();
boost::uint64_t getLocalTime();
boost::uint64_t getUtcTimeFromLocal(boost::uint64_t local);
boost::uint64_t getLocalTimeFromUtc(boost::uint64_t utc);

boost::uint64_t getFastMonoClock() NOEXCEPT;
double getHiResMonoClock() NOEXCEPT;

struct DateTime {
	unsigned yr;
	unsigned mon;
	unsigned day;

	unsigned hr;
	unsigned min;
	unsigned sec;

	unsigned ms;
};

DateTime breakDownTime(boost::uint64_t ms);
boost::uint64_t assembleTime(const DateTime &dt);

std::size_t formatTime(char *buffer, std::size_t max, boost::uint64_t ms, bool showMs);
boost::uint64_t scanTime(const char *str);

}

#endif
