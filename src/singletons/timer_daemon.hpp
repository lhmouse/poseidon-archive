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
	static void start();
	static void stop();

	// 如无特殊说明，时间单位一律用毫秒。
	// 返回的 shared_ptr 是该计时器的唯一持有者。

	// period 填零表示只触发一次。
	static boost::shared_ptr<TimerItem> registerAbsoluteTimer(
		boost::uint64_t timePoint, // 用 getFastMonoClock() 作参考。
		boost::uint64_t period, TimerCallback callback);
	static boost::shared_ptr<TimerItem> registerTimer(
		boost::uint64_t first, boost::uint64_t period, TimerCallback callback);

	static boost::shared_ptr<TimerItem> registerHourlyTimer(
		unsigned minute, unsigned second, TimerCallback callback);
	static boost::shared_ptr<TimerItem> registerDailyTimer(
		unsigned hour, unsigned minute, unsigned second, TimerCallback callback);
	// 0 = 星期日
	static boost::shared_ptr<TimerItem> registerWeeklyTimer(
		unsigned dayOfWeek, unsigned hour, unsigned minute, unsigned second, TimerCallback callback);

	// 直接在计时器线程中调用，不走消息队列。
	static boost::shared_ptr<TimerItem> registerLowLevelAbsoluteTimer(
		boost::uint64_t timePoint, // 用 getFastMonoClock() 作参考。
		boost::uint64_t period, TimerCallback callback);
	static boost::shared_ptr<TimerItem> registerLowLevelTimer(
		boost::uint64_t first, boost::uint64_t period, TimerCallback callback);

private:
	TimerDaemon();
};

}

#endif
