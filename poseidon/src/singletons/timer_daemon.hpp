// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2018, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_SINGLETONS_TIMER_DAEMON_HPP_
#define POSEIDON_SINGLETONS_TIMER_DAEMON_HPP_

#include <boost/shared_ptr.hpp>
#include <boost/function.hpp>
#include <boost/cstdint.hpp>

namespace Poseidon {

class Timer; // 没有定义的类，当作句柄使用。

class Timer_daemon {
private:
	Timer_daemon();

public:
	enum {
		period_intact  = -1ull,
	};

	typedef std::function<void (const boost::shared_ptr<Timer> &item, std::uint64_t now, std::uint64_t period)> Timer_callback;

	static void start();
	static void stop();

	// 时间单位一律用毫秒。
	// 返回的 shared_ptr 是该计时器的唯一持有者。

	// first 用 get_fast_mono_clock() 作参考，period 填零表示只触发一次。
	static boost::shared_ptr<Timer> register_absolute_timer(std::uint64_t first, std::uint64_t period, Timer_callback callback);
	static boost::shared_ptr<Timer> register_timer(std::uint64_t delta_first, std::uint64_t period, Timer_callback callback);

	static boost::shared_ptr<Timer> register_hourly_timer(unsigned minute, unsigned second, Timer_callback callback, bool utc);
	static boost::shared_ptr<Timer> register_daily_timer(unsigned hour, unsigned minute, unsigned second, Timer_callback callback, bool utc);
	// 0 = 星期日
	static boost::shared_ptr<Timer> register_weekly_timer(unsigned day_of_week, unsigned hour, unsigned minute, unsigned second, Timer_callback callback, bool utc);

	static boost::shared_ptr<Timer> register_low_level_absolute_timer(std::uint64_t first, std::uint64_t period, Timer_callback callback);
	static boost::shared_ptr<Timer> register_low_level_timer(std::uint64_t delta_first, std::uint64_t period, Timer_callback callback);

	static void set_absolute_time(const boost::shared_ptr<Timer> &item, std::uint64_t first, std::uint64_t period = period_intact);
	static void set_time(const boost::shared_ptr<Timer> &item, std::uint64_t delta_first, std::uint64_t period = period_intact);
};

}

#endif
