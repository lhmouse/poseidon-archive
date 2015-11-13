// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2015, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_TIME_HPP_
#define POSEIDON_TIME_HPP_

#include "cxx_ver.hpp"
#include <boost/cstdint.hpp>

namespace Poseidon {

// 时间单位是毫秒。

boost::uint64_t get_utc_time();
boost::uint64_t get_local_time();
boost::uint64_t get_utc_time_from_local(boost::uint64_t local);
boost::uint64_t get_local_time_from_utc(boost::uint64_t utc);

boost::uint64_t get_fast_mono_clock() NOEXCEPT;
double get_hi_res_mono_clock() NOEXCEPT;

struct DateTime {
	unsigned yr;
	unsigned mon;
	unsigned day;

	unsigned hr;
	unsigned min;
	unsigned sec;

	unsigned ms;
};

DateTime break_down_time(boost::uint64_t ms);
boost::uint64_t assemble_time(const DateTime &dt);

std::size_t format_time(char *buffer, std::size_t max, boost::uint64_t ms, bool show_ms);
boost::uint64_t scan_time(const char *str);

}

#endif
