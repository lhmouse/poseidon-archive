// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2018, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_TIME_HPP_
#define POSEIDON_TIME_HPP_

#include "cxx_ver.hpp"
#include <boost/cstdint.hpp>

namespace Poseidon {

// 时间单位是毫秒。

extern std::uint64_t get_utc_time();
extern std::uint64_t get_local_time();
extern std::uint64_t get_utc_time_from_local(std::uint64_t local);
extern std::uint64_t get_local_time_from_utc(std::uint64_t utc);

extern std::uint64_t get_fast_mono_clock() NOEXCEPT;
extern double get_hi_res_mono_clock() NOEXCEPT;

struct Date_time {
	unsigned yr;
	unsigned mon;
	unsigned day;

	unsigned hr;
	unsigned min;
	unsigned sec;

	unsigned ms;
};

extern Date_time break_down_time(std::uint64_t ms);
extern std::uint64_t assemble_time(const Date_time &dt);

extern std::size_t format_time(char *buffer, std::size_t max, std::uint64_t ms, bool show_ms);
extern std::uint64_t scan_time(const char *str);

}

#endif
