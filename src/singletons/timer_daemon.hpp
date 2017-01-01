// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2017, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_SINGLETONS_TIMER_DAEMON_HPP_
#define POSEIDON_SINGLETONS_TIMER_DAEMON_HPP_

#include "../cxx_ver.hpp"
#include <boost/shared_ptr.hpp>
#include <boost/function.hpp>
#include <boost/cstdint.hpp>

namespace Poseidon {

class TimerItem;

typedef boost::function<
	void (const boost::shared_ptr<TimerItem> &item, boost::uint64_t now, boost::uint64_t period)
	> TimerCallback;

class TimerDaemon {
private:
	TimerDaemon();

public:
	enum {
		PERIOD_NOT_MODIFIED =   (boost::uint64_t)-1,
	};

	static void start();
	static void stop();

	// 时间单位一律用毫秒。
	// 返回的 shared_ptr 是该计时器的唯一持有者。

	// time_point 用 get_fast_mono_clock() 作参考，period 填零表示只触发一次。
	static boost::shared_ptr<TimerItem> register_absolute_timer(
		boost::uint64_t time_point, boost::uint64_t period, TimerCallback callback);
	static boost::shared_ptr<TimerItem> register_timer(
		boost::uint64_t first, boost::uint64_t period, TimerCallback callback);

	static boost::shared_ptr<TimerItem> register_hourly_timer(
		unsigned minute, unsigned second, TimerCallback callback);
	static boost::shared_ptr<TimerItem> register_daily_timer(
		unsigned hour, unsigned minute, unsigned second, TimerCallback callback);
	// 0 = 星期日
	static boost::shared_ptr<TimerItem> register_weekly_timer(
		unsigned day_of_week, unsigned hour, unsigned minute, unsigned second, TimerCallback callback);

	static boost::shared_ptr<TimerItem> register_low_level_absolute_timer(
		boost::uint64_t time_point, boost::uint64_t period, TimerCallback callback);

	static void set_absolute_time(const boost::shared_ptr<TimerItem> &item,
		boost::uint64_t time_point, boost::uint64_t period = PERIOD_NOT_MODIFIED);
	static void set_time(const boost::shared_ptr<TimerItem> &item,
		boost::uint64_t first, boost::uint64_t period = PERIOD_NOT_MODIFIED);
};

}

#endif
