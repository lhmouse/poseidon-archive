// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2015, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_SINGLETONS_TIMER_DAEMON_HPP_
#define POSEIDON_SINGLETONS_TIMER_DAEMON_HPP_

#include "../cxx_ver.hpp"
#include <boost/shared_ptr.hpp>
#include <boost/function.hpp>
#include <boost/cstdint.hpp>

namespace Poseidon {

class TimerItem;

typedef boost::function<
	void (boost::uint64_t now, boost::uint64_t period)
	> TimerCallback;

struct TimerDaemon {
	enum {
		PERIOD_NOT_MODIFIED =	(boost::uint64_t)-1,
	};

	static void start();
	static void stop();

	// 时间单位一律用毫秒。
	// 返回的 shared_ptr 是该计时器的唯一持有者。

	// period 填零表示只触发一次。
	static boost::shared_ptr<TimerItem> registerAbsoluteTimer(
		boost::uint64_t timePoint, // 用 getFastMonoClock() 作参考。
		boost::uint64_t period, TimerCallback callback, bool isLowLevel = false);
	static boost::shared_ptr<TimerItem> registerTimer(
		boost::uint64_t first, boost::uint64_t period, TimerCallback callback, bool isLowLevel = false);

	static boost::shared_ptr<TimerItem> registerHourlyTimer(
		unsigned minute, unsigned second, TimerCallback callback, bool isLowLevel = false);
	static boost::shared_ptr<TimerItem> registerDailyTimer(
		unsigned hour, unsigned minute, unsigned second, TimerCallback callback, bool isLowLevel = false);
	// 0 = 星期日
	static boost::shared_ptr<TimerItem> registerWeeklyTimer(
		unsigned dayOfWeek, unsigned hour, unsigned minute, unsigned second, TimerCallback callback, bool isLowLevel = false);

	static void setAbsoluteTime(const boost::shared_ptr<TimerItem> &item,
		boost::uint64_t timePoint, boost::uint64_t period = PERIOD_NOT_MODIFIED);
	static void setTime(const boost::shared_ptr<TimerItem> &item,
		boost::uint64_t first, boost::uint64_t period = PERIOD_NOT_MODIFIED);

private:
	TimerDaemon();
};

}

#endif
