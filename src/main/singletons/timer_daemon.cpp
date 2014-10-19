#include "../../precompiled.hpp"
#include "timer_daemon.hpp"
#include <vector>
#include <algorithm>
#include <boost/thread.hpp>
#include <boost/shared_ptr.hpp>
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

class Poseidon::TimerItem : boost::noncopyable,
	public boost::enable_shared_from_this<TimerItem>
{
private:
	const unsigned long long m_period;
	const boost::weak_ptr<const void> m_dependency;
	const TimerCallback m_callback;

public:
	TimerItem(unsigned long long period,
		const boost::weak_ptr<const void> &dependency, const TimerCallback &callback)
		: m_period(period), m_dependency(dependency), m_callback(callback)
	{
	}
	~TimerItem(){
		LOG_INFO("Destroyed a timer item which has a period of ", m_period, " microsecond(s).");
	}

public:
	boost::shared_ptr<const TimerCallback>
		lock(boost::shared_ptr<const void> &lockedDep) const
	{
		if((m_dependency < boost::weak_ptr<void>()) || (boost::weak_ptr<void>() < m_dependency)){
			lockedDep = m_dependency.lock();
			if(!lockedDep){
				return NULLPTR;
			}
		}
		return boost::shared_ptr<const TimerCallback>(shared_from_this(), &m_callback);
	}

	unsigned long long getPeriod() const {
		return m_period;
	}
};

namespace {

const unsigned long long MILLISECS_PER_HOUR	= 3600 * 1000;
const unsigned long long MILLISECS_PER_DAY	= 24 * MILLISECS_PER_HOUR;
const unsigned long long MILLISECS_PER_WEEK	= 7 * MILLISECS_PER_DAY;

class TimerJob : public JobBase {
private:
	const boost::shared_ptr<const void> m_dependency;
	const boost::shared_ptr<const TimerCallback> m_callback;
	const unsigned long long m_now;
	const unsigned long long m_period;

public:
	TimerJob(boost::shared_ptr<const void> dependency,
		boost::shared_ptr<const TimerCallback> callback,
		unsigned long long now, unsigned long long period)
		: m_dependency(STD_MOVE(dependency))
		, m_callback(STD_MOVE(callback))
		, m_now(now), m_period(period)
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

		boost::shared_ptr<const void> lockedDep;
		boost::shared_ptr<const TimerCallback> callback;
		unsigned long long period = 0;
		{
			const boost::mutex::scoped_lock lock(g_mutex);
			while(!g_timers.empty() && (now >= g_timers.front().next)){
				const AUTO(item, g_timers.front().item.lock());
				std::pop_heap(g_timers.begin(), g_timers.end());
				if(item){
					callback = item->lock(lockedDep);
					if(callback){
						period = item->getPeriod();
						if(period == 0){
							g_timers.pop_back();
						} else {
							g_timers.back().next += period;
							std::push_heap(g_timers.begin(), g_timers.end());
						}
						break;
					}
				}
				g_timers.pop_back();
			}
		}
		if(!callback){
			lockedDep.reset();
			::usleep(100000);
			continue;
		}

		try {
			LOG_INFO("Preparing a timer job for dispatching.");
			boost::make_shared<TimerJob>(
				STD_MOVE(lockedDep), STD_MOVE(callback), now, period)->pend();
		} catch(std::exception &e){
			LOG_ERROR("std::exception thrown while dispatching timer job, what = ", e.what());
		} catch(...){
			LOG_ERROR("Unknown exception thrown while dispatching timer job.");
		}
	}
}

void threadProc(){
	PROFILE_ME;
	Logger::setThreadTag(Logger::TAG_TIMER);
	LOG_INFO("Timer daemon started.");

	daemonLoop();

	LOG_INFO("Timer daemon stopped.");
}

}

boost::shared_ptr<const TimerItem> TimerDaemon::registerAbsoluteTimer(
	unsigned long long timePoint, unsigned long long period,
	const boost::weak_ptr<const void> &dependency, const TimerCallback &callback)
{
	AUTO(item, boost::make_shared<TimerItem>(
		period * 1000, boost::ref(dependency), boost::ref(callback)));
	TimerQueueElement tqe;
	tqe.next = timePoint;
	tqe.item = item;
	{
		const boost::mutex::scoped_lock lock(g_mutex);
		g_timers.push_back(tqe);
		std::push_heap(g_timers.begin(), g_timers.end());
	}
	LOG_INFO("Created a timer item which will be triggered ",
		std::max<long long>(0, timePoint - getMonoClock()),
		" microsecond(s) later and has a period of ", item->getPeriod(), " microsecond(s).");
	return item;
}
boost::shared_ptr<const TimerItem> TimerDaemon::registerTimer(
	unsigned long long first, unsigned long long period,
	const boost::weak_ptr<const void> &dependency, const TimerCallback &callback)
{
	return registerAbsoluteTimer(
		getMonoClock() + first * 1000, period, dependency, callback);
}

boost::shared_ptr<const TimerItem> TimerDaemon::registerHourlyTimer(
	unsigned minute, unsigned second,
	const boost::weak_ptr<const void> &dependency, const TimerCallback &callback)
{
	const unsigned long long delta = getLocalTime() -
		(minute * 60ull + second) * 1000;
	return registerTimer(
		MILLISECS_PER_HOUR - delta % MILLISECS_PER_HOUR, MILLISECS_PER_HOUR,
		dependency, callback);
}
boost::shared_ptr<const TimerItem> TimerDaemon::registerDailyTimer(
	unsigned hour, unsigned minute, unsigned second,
	const boost::weak_ptr<const void> &dependency, const TimerCallback &callback)
{
	const unsigned long long delta = getLocalTime() -
		(hour * 3600ull + minute * 60ull + second) * 1000;
	return registerTimer(
		MILLISECS_PER_DAY - delta % MILLISECS_PER_DAY, MILLISECS_PER_DAY,
		dependency, callback);
}
boost::shared_ptr<const TimerItem> TimerDaemon::registerWeeklyTimer(
	unsigned dayOfWeek, unsigned hour, unsigned minute, unsigned second,
	const boost::weak_ptr<const void> &dependency, const TimerCallback &callback)
{
	// 注意 1970-01-01 是星期四。
	const unsigned long long delta = getLocalTime() -
		((dayOfWeek + 3) * 86400ull + hour * 3600ull + minute * 60ull + second) * 1000;
	return registerTimer(
		MILLISECS_PER_WEEK - delta % MILLISECS_PER_WEEK, MILLISECS_PER_WEEK,
		dependency, callback);
}

void TimerDaemon::start(){
	if(atomicExchange(g_running, true) != false){
		LOG_FATAL("Only one daemon is allowed at the same time.");
		std::abort();
	}
	LOG_INFO("Starting timer daemon...");

	boost::thread(threadProc).swap(g_thread);
}
void TimerDaemon::stop(){
	LOG_INFO("Stopping timer daemon...");

	atomicStore(g_running, false);
	if(g_thread.joinable()){
		g_thread.join();
	}
	g_timers.clear();
}
