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
#include "job_dispatcher.hpp"
using namespace Poseidon;

namespace {

const unsigned long long MILLISECS_PER_HOUR	= 3600 * 1000;
const unsigned long long MILLISECS_PER_DAY	= 24 * MILLISECS_PER_HOUR;
const unsigned long long MILLISECS_PER_WEEK	= 7 * MILLISECS_PER_DAY;

class TimerItem : public JobBase {
private:
	const unsigned long long m_period;

	boost::function<void ()> m_callback;
	unsigned long long m_next;

public:
	TimerItem(unsigned long long period,
		boost::function<void ()> &callback, unsigned long long next)
		: m_period(period), m_next(next)
	{
		m_callback.swap(callback);
	}

protected:
	virtual void perform() const {
		m_callback();
	}

public:
	bool isLaterThan(unsigned long long time) const {
		return m_next > time;
	}
	bool isLaterThan(const TimerItem &rhs) const {
		return m_next > rhs.m_next;
	}
	bool increment(){
		if(m_period == 0){
			return false;
		}
		m_next += m_period;
		return true;
	}
};

bool timerItemComp(const boost::shared_ptr<TimerItem> &lhs,
	const boost::shared_ptr<TimerItem> &rhs)
{
	return lhs->isLaterThan(*rhs);
}

volatile bool g_daemonRunning = false;
boost::thread g_daemonThread;

boost::mutex g_mutex;
std::vector<boost::shared_ptr<TimerItem> > g_timerHeap;

void registerTimerUsingMove(unsigned long long period,
	boost::function<void ()> &callback, unsigned long long first)
{
	AUTO(item, boost::make_shared<TimerItem>(
		period * 1000, boost::ref(callback), getMonoClock() + first * 1000
	));
	{
		const boost::mutex::scoped_lock lock(g_mutex);
		g_timerHeap.push_back(item);
		std::push_heap(g_timerHeap.begin(), g_timerHeap.end(), timerItemComp);
	}

	LOG_INFO <<"Registered timer with period "
		<<period <<" millisecond(s) and will be triggered after "
		<<first <<" millisecond(s).";
}

void threadProc(){
	while(atomicLoad(g_daemonRunning)){
		const unsigned long long now = getMonoClock();
		boost::shared_ptr<TimerItem> ti;
		{
			const boost::mutex::scoped_lock lock(g_mutex);
			if(!g_timerHeap.empty() && !g_timerHeap.front()->isLaterThan(now)){
				std::pop_heap(g_timerHeap.begin(), g_timerHeap.end(), timerItemComp);
				ti = g_timerHeap.back();
				if(ti->increment()){
					std::push_heap(g_timerHeap.begin(), g_timerHeap.end(), timerItemComp);
				} else {
					g_timerHeap.pop_back();
				}
			}
		}
		if(!ti){
			::sleep(1);
			continue;
		}
		try {
			LOG_INFO <<"Preparing a timer job for dispatching.";
			ti->pend();
		} catch(std::exception &e){
			LOG_ERROR <<"std::exception thrown while dispatching timer job, what = " <<e.what();
		} catch(...){
			LOG_ERROR <<"Unknown exception thrown while dispatching timer job.";
		}
	}
}

}

void TimerDaemon::start(){
	if(atomicExchange(g_daemonRunning, true) != false){
		LOG_FATAL <<"Only one daemon is allowed at the same time.";
		std::abort();
	}
	LOG_INFO <<"Starting timer daemon...";

	boost::thread(threadProc).swap(g_daemonThread);
}
void TimerDaemon::stop(){
	LOG_INFO <<"Stopping timer daemon...";

	atomicStore(g_daemonRunning, false);
	if(g_daemonThread.joinable()){
		g_daemonThread.join();
	}
	g_timerHeap.clear();
}

void TimerDaemon::registerTimer(boost::function<void ()> callback,
	unsigned long long period, unsigned long long first)
{
	registerTimerUsingMove(period, callback, first);
}
void TimerDaemon::registerHourlyTimer(boost::function<void ()> callback,
	unsigned minute, unsigned second)
{
	const unsigned long long delta = getLocalTime() -
		(minute * 60ull + second) * 1000;
	registerTimerUsingMove(MILLISECS_PER_HOUR, callback,
		MILLISECS_PER_HOUR - delta % MILLISECS_PER_HOUR);
}
void TimerDaemon::registerDailyTimer(boost::function<void ()> callback,
	unsigned hour, unsigned minute, unsigned second)
{
	const unsigned long long delta = getLocalTime() -
		(hour * 3600ull + minute * 60ull + second) * 1000;
	registerTimerUsingMove(MILLISECS_PER_DAY, callback,
		MILLISECS_PER_DAY - delta % MILLISECS_PER_DAY);
}
void TimerDaemon::registerWeeklyTimer(boost::function<void ()> callback,
	unsigned dayOfWeek, unsigned hour, unsigned minute, unsigned second)
{
	// 注意 1970-01-01 是星期四。
	const unsigned long long delta = getLocalTime() -
		((dayOfWeek + 3) * 86400ull + hour * 3600ull + minute * 60ull + second) * 1000;
	registerTimerUsingMove(MILLISECS_PER_WEEK, callback,
		MILLISECS_PER_WEEK - delta % MILLISECS_PER_WEEK);
}
