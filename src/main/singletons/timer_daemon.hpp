// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_SINGLETONS_TIMER_DAEMON_HPP_
#define POSEIDON_SINGLETONS_TIMER_DAEMON_HPP_

#include "../cxx_ver.hpp"
#include <boost/shared_ptr.hpp>
#include <boost/function.hpp>

namespace Poseidon {

class TimerItem;

typedef boost::function<
	void (unsigned long long now, unsigned long long period)
	> TimerCallback;

struct TimerDaemon {
	static void start();
	static void stop();

	// 如无特殊说明，时间单位一律用毫秒。
	// 返回的 shared_ptr 是该计时器的唯一持有者。

	// period 填零表示只触发一次。
	static boost::shared_ptr<TimerItem> registerAbsoluteTimer(
		unsigned long long timePoint, // 用 getMonoClock() 作参考，单位微秒。
		unsigned long long period, TimerCallback callback);
	static boost::shared_ptr<TimerItem> registerTimer(
		unsigned long long first, unsigned long long period, TimerCallback callback);

	static boost::shared_ptr<TimerItem> registerHourlyTimer(
		unsigned minute, unsigned second, TimerCallback callback);
	static boost::shared_ptr<TimerItem> registerDailyTimer(
		unsigned hour, unsigned minute, unsigned second, TimerCallback callback);
	// 0 = 星期日
	static boost::shared_ptr<TimerItem> registerWeeklyTimer(
		unsigned dayOfWeek, unsigned hour, unsigned minute, unsigned second, TimerCallback callback);

private:
	TimerDaemon();
};

}

#endif
