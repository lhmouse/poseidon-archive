// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014, LH_Mouse. All wrongs reserved.

#include "../precompiled.hpp"
#include "timer_daemon.hpp"
#include <vector>
#include <algorithm>
#include <boost/thread.hpp>
#include <boost/make_shared.hpp>
#include <boost/ref.hpp>
#include <unistd.h>
#include "../log.hpp"
#include "../atomic.hpp"
#include "../exception.hpp"
#include "../utilities.hpp"
#include "../job_base.hpp"
#include "../profiler.hpp"
using namespace Poseidon;

struct Poseidon::TimerItem : NONCOPYABLE {
	const unsigned long long period;
	const boost::shared_ptr<const TimerCallback> callback;

	TimerItem(unsigned long long period_, boost::shared_ptr<const TimerCallback> callback_)
		: period(period_), callback(STD_MOVE(callback_))
	{
		LOG_POSEIDON_INFO("Created timer, period = ", period);
	}
	~TimerItem(){
		LOG_POSEIDON_INFO("Destroyed timer, period = ", period);
	}
};

namespace {

const unsigned long long MILLISECS_PER_HOUR	= 3600 * 1000;
const unsigned long long MILLISECS_PER_DAY	= 24 * MILLISECS_PER_HOUR;
const unsigned long long MILLISECS_PER_WEEK	= 7 * MILLISECS_PER_DAY;

class TimerJob : public JobBase {
private:
	const boost::shared_ptr<const TimerCallback> m_callback;
	const unsigned long long m_now;
	const unsigned long long m_period;

public:
	TimerJob(boost::shared_ptr<const TimerCallback> callback,
		unsigned long long now, unsigned long long period)
		: m_callback(STD_MOVE(callback)), m_now(now), m_period(period)
	{
	}

public:
	void perform(){
		PROFILE_ME;

		(*m_callback)(m_now, m_period);
	}
};

struct TimerQueueElement {
	unsigned long long next;
	boost::weak_ptr<TimerItem> item;

	bool operator<(const TimerQueueElement &rhs) const {
		return next > rhs.next;
	}
};

volatile bool g_running = false;
boost::thread g_thread;

boost::mutex g_mutex;
std::vector<TimerQueueElement> g_timers;

void daemonLoop(){
	while(atomicLoad(g_running)){
		const unsigned long long now = getMonoClock();

		boost::shared_ptr<const TimerCallback> callback;
		unsigned long long period = 0;
		{
			const boost::mutex::scoped_lock lock(g_mutex);
			while(!g_timers.empty() && (now >= g_timers.front().next)){
				const AUTO(item, g_timers.front().item.lock());
				std::pop_heap(g_timers.begin(), g_timers.end());
				if(item){
					callback = item->callback;
					period = item->period;
					if(period == 0){
						g_timers.pop_back();
					} else {
						g_timers.back().next += period;
						std::push_heap(g_timers.begin(), g_timers.end());
					}
					break;
				}
				g_timers.pop_back();
			}
		}
		if(!callback){
			::usleep(100000);
			continue;
		}

		try {
			LOG_POSEIDON_TRACE("Preparing a timer job for dispatching: period = ", period);
			pendJob(boost::make_shared<TimerJob>(STD_MOVE(callback), now, period));
		} catch(std::exception &e){
			LOG_POSEIDON_ERROR("std::exception thrown while dispatching timer job, what = ", e.what());
		} catch(...){
			LOG_POSEIDON_ERROR("Unknown exception thrown while dispatching timer job.");
		}
	}
}

void threadProc(){
	PROFILE_ME;
	Logger::setThreadTag("  T "); // Timer
	LOG_POSEIDON_INFO("Timer daemon started.");

	daemonLoop();

	LOG_POSEIDON_INFO("Timer daemon stopped.");
}

}

void TimerDaemon::start(){
	if(atomicExchange(g_running, true) != false){
		LOG_POSEIDON_FATAL("Only one daemon is allowed at the same time.");
		std::abort();
	}
	LOG_POSEIDON_INFO("Starting timer daemon...");

	boost::thread(threadProc).swap(g_thread);
}
void TimerDaemon::stop(){
	if(atomicExchange(g_running, false) == false){
		return;
	}
	LOG_POSEIDON_INFO("Stopping timer daemon...");

	if(g_thread.joinable()){
		g_thread.join();
	}
	g_timers.clear();
}

boost::shared_ptr<TimerItem> TimerDaemon::registerAbsoluteTimer(
	unsigned long long timePoint, unsigned long long period, TimerCallback callback)
{
	AUTO(sharedCallback, boost::make_shared<TimerCallback>());
	sharedCallback->swap(callback);
	AUTO(item, boost::make_shared<TimerItem>(period * 1000, sharedCallback));

	TimerQueueElement tqe;
	tqe.next = timePoint;
	tqe.item = item;
	{
		const boost::mutex::scoped_lock lock(g_mutex);
		g_timers.push_back(tqe);
		std::push_heap(g_timers.begin(), g_timers.end());
	}
	LOG_POSEIDON_INFO("Created a timer item which will be triggered ",
		std::max<long long>(0, timePoint - getMonoClock()),
		" microsecond(s) later and has a period of ", item->period, " microsecond(s).");
	return item;
}
boost::shared_ptr<TimerItem> TimerDaemon::registerTimer(
	unsigned long long first, unsigned long long period, TimerCallback callback)
{
	return registerAbsoluteTimer(getMonoClock() + first * 1000, period, STD_MOVE(callback));
}

boost::shared_ptr<TimerItem> TimerDaemon::registerHourlyTimer(
	unsigned minute, unsigned second, TimerCallback callback)
{
	const unsigned long long delta = getLocalTime() -
		(minute * 60ull + second) * 1000;
	return registerTimer(MILLISECS_PER_HOUR - delta % MILLISECS_PER_HOUR, MILLISECS_PER_HOUR,
		STD_MOVE(callback));
}
boost::shared_ptr<TimerItem> TimerDaemon::registerDailyTimer(
	unsigned hour, unsigned minute, unsigned second, TimerCallback callback)
{
	const unsigned long long delta = getLocalTime() -
		(hour * 3600ull + minute * 60ull + second) * 1000;
	return registerTimer(MILLISECS_PER_DAY - delta % MILLISECS_PER_DAY, MILLISECS_PER_DAY,
		STD_MOVE(callback));
}
boost::shared_ptr<TimerItem> TimerDaemon::registerWeeklyTimer(
	unsigned dayOfWeek, unsigned hour, unsigned minute, unsigned second, TimerCallback callback)
{
	// 注意 1970-01-01 是星期四。
	const unsigned long long delta = getLocalTime() -
		((dayOfWeek + 3) * 86400ull + hour * 3600ull + minute * 60ull + second) * 1000;
	return registerTimer(MILLISECS_PER_WEEK - delta % MILLISECS_PER_WEEK, MILLISECS_PER_WEEK,
		STD_MOVE(callback));
}
