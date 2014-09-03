#include "../../precompiled.hpp"
#include "timer_manager.hpp"
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

const boost::weak_ptr<void> NULL_WEAK_PTR;

}

class Poseidon::TimerServlet : boost::noncopyable,
	public boost::enable_shared_from_this<TimerServlet>
{
private:
	const unsigned long long m_period;
	const boost::weak_ptr<void> m_dependency;
	const TimerCallback m_callback;

public:
	TimerServlet(unsigned long long period,
		const boost::weak_ptr<void> &dependency, const TimerCallback &callback)
		: m_period(period), m_dependency(dependency), m_callback(callback)
	{
		LOG_INFO("Created timer servlet with period = ", m_period, " microseconds");
	}
	~TimerServlet(){
		LOG_INFO("Destroyed timer servlet with period = ", m_period, " microseconds");
	}

public:
    boost::shared_ptr<const TimerCallback> lock() const {
    	if(!(m_dependency < NULL_WEAK_PTR) && !(NULL_WEAK_PTR < m_dependency)){
    		return boost::shared_ptr<const TimerCallback>(shared_from_this(), &m_callback);
    	}
    	const AUTO(lockedDep, m_dependency.lock());
    	if(!lockedDep){
    		return boost::shared_ptr<const TimerCallback>();
    	}
    	return boost::shared_ptr<const TimerCallback>(
    		boost::make_shared<
    			std::pair<boost::shared_ptr<void>, boost::shared_ptr<const TimerServlet> >
    			>(lockedDep, shared_from_this()),
    		&m_callback
    	);
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
	const boost::shared_ptr<const TimerCallback> m_callback;
	const unsigned long long m_period;

public:
	TimerJob(boost::shared_ptr<const TimerCallback> callback, unsigned long long period)
		: m_callback(STD_MOVE(callback)), m_period(period)
	{
	}

public:
	void perform() const {
		(*m_callback)(m_period);
	}
};

struct TimerItem {
	unsigned long long next;
	boost::weak_ptr<TimerServlet> servlet;

	bool operator<(const TimerItem &rhs) const {
		return next > rhs.next;
	}
};

volatile bool g_daemonRunning = false;
boost::thread g_daemonThread;

boost::mutex g_mutex;
std::vector<TimerItem> g_timers;

void threadProc(){
	LOG_INFO("Timer daemon started.");

	while(atomicLoad(g_daemonRunning)){
		const unsigned long long now = getMonoClock();
		boost::shared_ptr<const TimerCallback> callback;
		unsigned long long period;
		{
			const boost::mutex::scoped_lock lock(g_mutex);
			while(!g_timers.empty() && (now >= g_timers.front().next)){
				const AUTO(servlet, g_timers.front().servlet.lock());
				std::pop_heap(g_timers.begin(), g_timers.end());
				if(servlet){
					callback = servlet->lock();
					if(callback){
						period = servlet->getPeriod();
						if(servlet->getPeriod() == 0){
							g_timers.pop_back();
						} else {
							g_timers.back().next += servlet->getPeriod();
							std::push_heap(g_timers.begin(), g_timers.end());
						}
						break;
					}
				}
				g_timers.pop_back();
			}
		}
		if(!callback){
			::sleep(1);
			continue;
		}
		try {
			LOG_INFO("Preparing a timer job for dispatching.");

			boost::make_shared<TimerJob>(STD_MOVE(callback), period)->pend();
		} catch(std::exception &e){
			LOG_ERROR("std::exception thrown while dispatching timer job, what = ", e.what());
		} catch(...){
			LOG_ERROR("Unknown exception thrown while dispatching timer job.");
		}
	}

	LOG_INFO("Timer daemon stopped.");
}

}

boost::shared_ptr<const TimerServlet> TimerManager::registerTimer(
	unsigned long long first, unsigned long long period,
	const boost::weak_ptr<void> &dependency, const TimerCallback &callback)
{
	AUTO(servlet, boost::make_shared<TimerServlet>(period * 1000,
		boost::ref(dependency), boost::ref(callback)));
	TimerItem ti;
	ti.next = getMonoClock() + first * 1000;
	ti.servlet = servlet;
	{
		const boost::mutex::scoped_lock lock(g_mutex);
		g_timers.push_back(ti);
		std::push_heap(g_timers.begin(), g_timers.end());
	}
	return servlet;
}

boost::shared_ptr<const TimerServlet> TimerManager::registerHourlyTimer(
	unsigned minute, unsigned second,
	const boost::weak_ptr<void> &dependency, const TimerCallback &callback)
{
	const unsigned long long delta = getLocalTime() -
		(minute * 60ull + second) * 1000;
	return registerTimer(
		MILLISECS_PER_HOUR - delta % MILLISECS_PER_HOUR, MILLISECS_PER_HOUR,
		dependency, callback);
}
boost::shared_ptr<const TimerServlet> TimerManager::registerDailyTimer(
	unsigned hour, unsigned minute, unsigned second,
	const boost::weak_ptr<void> &dependency, const TimerCallback &callback)
{
	const unsigned long long delta = getLocalTime() -
		(hour * 3600ull + minute * 60ull + second) * 1000;
	return registerTimer(
		MILLISECS_PER_DAY - delta % MILLISECS_PER_DAY, MILLISECS_PER_DAY,
		dependency, callback);
}
boost::shared_ptr<const TimerServlet> TimerManager::registerWeeklyTimer(
	unsigned dayOfWeek, unsigned hour, unsigned minute, unsigned second,
	const boost::weak_ptr<void> &dependency, const TimerCallback &callback)
{
	// 注意 1970-01-01 是星期四。
	const unsigned long long delta = getLocalTime() -
		((dayOfWeek + 3) * 86400ull + hour * 3600ull + minute * 60ull + second) * 1000;
	return registerTimer(
		MILLISECS_PER_WEEK - delta % MILLISECS_PER_WEEK, MILLISECS_PER_WEEK,
		dependency, callback);
}

void TimerDaemon::start(){
	if(atomicExchange(g_daemonRunning, true) != false){
		LOG_FATAL("Only one daemon is allowed at the same time.");
		std::abort();
	}
	LOG_INFO("Starting timer daemon...");

	boost::thread(threadProc).swap(g_daemonThread);
}
void TimerDaemon::stop(){
	LOG_INFO("Stopping timer daemon...");

	atomicStore(g_daemonRunning, false);
	if(g_daemonThread.joinable()){
		g_daemonThread.join();
	}
	g_timers.clear();
}
