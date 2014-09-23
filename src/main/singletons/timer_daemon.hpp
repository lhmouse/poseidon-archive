#ifndef POSEIDON_SINGLETONS_TIMER_DAEMON_HPP_
#define POSEIDON_SINGLETONS_TIMER_DAEMON_HPP_

#include <boost/shared_ptr.hpp>
#include <boost/weak_ptr.hpp>
#include <boost/function.hpp>

namespace Poseidon {

class TimerItem;

typedef boost::function<
	void (unsigned long long period)
	> TimerCallback;

struct TimerDaemon {
	static void start();
	static void stop();

	// 时间单位一律用毫秒。
	// 返回的 shared_ptr 是该计时器的唯一持有者。
	// callback 禁止 move，否则可能出现主模块中引用子模块内存的情况。

	// period 填零表示只触发一次。
	static boost::shared_ptr<const TimerItem> registerTimer(
		unsigned long long first, unsigned long long period,
		const boost::weak_ptr<void> &dependency, const TimerCallback &callback);
	static boost::shared_ptr<const TimerItem> registerAbsoluteTimer(
		unsigned long long timePoint,
		const boost::weak_ptr<void> &dependency, const TimerCallback &callback);

	static boost::shared_ptr<const TimerItem> registerHourlyTimer(
		unsigned minute, unsigned second,
		const boost::weak_ptr<void> &dependency, const TimerCallback &callback);
	static boost::shared_ptr<const TimerItem> registerDailyTimer(
		unsigned hour, unsigned minute, unsigned second,
		const boost::weak_ptr<void> &dependency, const TimerCallback &callback);
	// 0 = 星期日
	static boost::shared_ptr<const TimerItem> registerWeeklyTimer(
		unsigned dayOfWeek, unsigned hour, unsigned minute, unsigned second,
		const boost::weak_ptr<void> &dependency, const TimerCallback &callback);

private:
	TimerDaemon();
};

}

#endif
