#ifndef POSEIDON_SINGLETONS_TIMER_DAEMON_HPP_
#define POSEIDON_SINGLETONS_TIMER_DAEMON_HPP_

#include "../../cxx_ver.hpp"
#include <boost/shared_ptr.hpp>
#include <boost/weak_ptr.hpp>

#ifdef POSEIDON_CXX11
#	include <functional>
#else
#	include <tr1/functional>
#endif

namespace Poseidon {

class TimerItem;

typedef TR1::function<
	void (unsigned long long now, unsigned long long period)
	> TimerCallback;

struct TimerDaemon {
	static void start();
	static void stop();

	// 如无特殊说明，时间单位一律用毫秒。
	// 返回的 shared_ptr 是该计时器的唯一持有者。
	// callback 禁止 move，否则可能出现主模块中引用子模块内存的情况。

	// period 填零表示只触发一次。
	static boost::shared_ptr<TimerItem> registerAbsoluteTimer(
		unsigned long long timePoint, // 用 getMonoClock() 作参考，单位微秒。
		unsigned long long period,
		const boost::weak_ptr<const void> &dependency, const TimerCallback &callback);
	static boost::shared_ptr<TimerItem> registerTimer(
		unsigned long long first, unsigned long long period,
		const boost::weak_ptr<const void> &dependency, const TimerCallback &callback);

	static boost::shared_ptr<TimerItem> registerHourlyTimer(
		unsigned minute, unsigned second,
		const boost::weak_ptr<const void> &dependency, const TimerCallback &callback);
	static boost::shared_ptr<TimerItem> registerDailyTimer(
		unsigned hour, unsigned minute, unsigned second,
		const boost::weak_ptr<const void> &dependency, const TimerCallback &callback);
	// 0 = 星期日
	static boost::shared_ptr<TimerItem> registerWeeklyTimer(
		unsigned dayOfWeek, unsigned hour, unsigned minute, unsigned second,
		const boost::weak_ptr<const void> &dependency, const TimerCallback &callback);

private:
	TimerDaemon();
};

}

#endif
