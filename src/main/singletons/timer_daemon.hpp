#ifndef POSEIDON_TIMER_DAEMON_HPP_
#define POSEIDON_TIMER_DAEMON_HPP_

#include <boost/function.hpp>

namespace Poseidon {

struct TimerDaemon {
	static void start();
	static void stop();

	// 时间单位一律用毫秒。

	// period 填零表示只触发一次。
	static void registerTimer(boost::function<void ()> callback,
		unsigned long long first, unsigned long long period);

	static void registerHourlyTimer(boost::function<void ()> callback,
		unsigned minute, unsigned second = 0);
	static void registerDailyTimer(boost::function<void ()> callback,
		unsigned hour, unsigned minute = 0, unsigned second = 0);
	// 0 = 星期日
	static void registerWeeklyTimer(boost::function<void ()> callback,
		unsigned dayOfWeek, unsigned hour, unsigned minute = 0, unsigned second = 0);

private:
	TimerDaemon();
};

}

#endif
