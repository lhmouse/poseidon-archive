// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2015, LH_Mouse. All wrongs reserved.

#include "../precompiled.hpp"
#include "timer_daemon.hpp"
#include "job_dispatcher.hpp"
#include "../thread.hpp"
#include "../log.hpp"
#include "../atomic.hpp"
#include "../exception.hpp"
#include "../mutex.hpp"
#include "../condition_variable.hpp"
#include "../time.hpp"
#include "../job_base.hpp"
#include "../profiler.hpp"

namespace Poseidon {

struct TimerItem : NONCOPYABLE {
	boost::uint64_t period;
	boost::shared_ptr<const TimerCallback> callback;
	bool isAsync;
	unsigned long stamp;

	TimerItem(boost::uint64_t period_, boost::shared_ptr<const TimerCallback> callback_, bool isAsync_)
		: period(period_), callback(STD_MOVE(callback_)), isAsync(isAsync_), stamp(0)
	{
		LOG_POSEIDON_DEBUG("Created timer: period = ", period, ", isAsync = ", isAsync);
	}
	~TimerItem(){
		LOG_POSEIDON_DEBUG("Destroyed timer: period = ", period, ", isAsync = ", isAsync);
	}
};

namespace {
	const boost::uint64_t MILLISECS_PER_HOUR	= 3600 * 1000;
	const boost::uint64_t MILLISECS_PER_DAY		= 24 * MILLISECS_PER_HOUR;
	const boost::uint64_t MILLISECS_PER_WEEK	= 7 * MILLISECS_PER_DAY;

	class TimerJob : public JobBase {
	private:
		const boost::weak_ptr<TimerItem> m_item;
		const boost::uint64_t m_now;

	public:
		TimerJob(boost::weak_ptr<TimerItem> item, boost::uint64_t now)
			: m_item(STD_MOVE(item)), m_now(now)
		{
		}

	public:
		boost::weak_ptr<const void> getCategory() const OVERRIDE {
			return m_item;
		}
		void perform() const OVERRIDE {
			PROFILE_ME;

			const AUTO(item, m_item.lock());
			if(!item){
				return;
			}

			(*item->callback)(item, m_now, item->period);
		}
	};

	struct TimerQueueElement {
		boost::uint64_t next;
		boost::weak_ptr<TimerItem> item;
		unsigned long stamp;

		TimerQueueElement(boost::uint64_t next_, boost::weak_ptr<TimerItem> item_, unsigned long stamp_)
			: next(next_), item(STD_MOVE(item_)), stamp(stamp_)
		{
		}

		bool operator<(const TimerQueueElement &rhs) const {
			return next > rhs.next;
		}
	};

	volatile bool g_running = false;
	Thread g_thread;

	Mutex g_mutex;
	ConditionVariable g_newTimer;
	std::vector<TimerQueueElement> g_timers;

	bool pumpOneElement() NOEXCEPT {
		const boost::uint64_t now = getFastMonoClock();

		boost::shared_ptr<TimerItem> item;
		{
			const Mutex::UniqueLock lock(g_mutex);
			while(!g_timers.empty() && (now >= g_timers.front().next)){
				item = g_timers.front().item.lock();
				const AUTO(stamp, g_timers.front().stamp);
				std::pop_heap(g_timers.begin(), g_timers.end());

				if(item && (stamp == item->stamp)){
					if(item->period == 0){
						g_timers.pop_back();
					} else {
						g_timers.back().next += item->period;
						std::push_heap(g_timers.begin(), g_timers.end());
					}
					break;
				}
				item.reset();
				g_timers.pop_back();
			}
		}
		if(!item){
			return false;
		}

		try {
			if(item->isAsync){
				LOG_POSEIDON_TRACE("Dispatching async timer");
				(*item->callback)(item, now, item->period);
			} else {
				LOG_POSEIDON_TRACE("Preparing a timer job for dispatching");
				enqueueJob(boost::make_shared<TimerJob>(item, now));
			}
		} catch(std::exception &e){
			LOG_POSEIDON_WARNING("std::exception thrown while dispatching timer job, what = ", e.what());
		} catch(...){
			LOG_POSEIDON_WARNING("Unknown exception thrown while dispatching timer job.");
		}

		return true;
	}

	void daemonLoop(){
		for(;;){
			while(pumpOneElement()){
				// noop
			}

			if(!atomicLoad(g_running, ATOMIC_CONSUME) || !JobDispatcher::isRunning()){
				break;
			}

			Mutex::UniqueLock lock(g_mutex);
			g_newTimer.timedWait(lock, 100);
		}
	}

	void threadProc(){
		PROFILE_ME;
		LOG_POSEIDON_INFO("Timer daemon started.");

		daemonLoop();

		LOG_POSEIDON_INFO("Timer daemon stopped.");
	}
}

void TimerDaemon::start(){
	if(atomicExchange(g_running, true, ATOMIC_ACQ_REL) != false){
		LOG_POSEIDON_FATAL("Only one daemon is allowed at the same time.");
		std::abort();
	}
	LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_INFO, "Starting timer daemon...");

	Thread(threadProc, "  T ").swap(g_thread);
}
void TimerDaemon::stop(){
	if(atomicExchange(g_running, false, ATOMIC_ACQ_REL) == false){
		return;
	}
	LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_INFO, "Stopping timer daemon...");

	if(g_thread.joinable()){
		g_thread.join();
	}
	g_timers.clear();
}

boost::shared_ptr<TimerItem> TimerDaemon::registerAbsoluteTimer(
	boost::uint64_t timePoint, boost::uint64_t period, TimerCallback callback, bool isAsync)
{
	PROFILE_ME;

	AUTO(item, boost::make_shared<TimerItem>(period, boost::make_shared<TimerCallback>(STD_MOVE(callback)), isAsync));
	{
		const Mutex::UniqueLock lock(g_mutex);
		g_timers.push_back(TimerQueueElement(timePoint, item, item->stamp));
		std::push_heap(g_timers.begin(), g_timers.end());
		g_newTimer.signal();
	}
	LOG_POSEIDON_DEBUG("Created a(n) ", isAsync ? "async " : "", "timer item which will be triggered ",
		std::max<boost::int64_t>(static_cast<boost::int64_t>(timePoint - getFastMonoClock()), 0),
		" microsecond(s) later and has a period of ", item->period, " microsecond(s).");
	return item;
}
boost::shared_ptr<TimerItem> TimerDaemon::registerTimer(
	boost::uint64_t first, boost::uint64_t period, TimerCallback callback, bool isAsync)
{
	return registerAbsoluteTimer(getFastMonoClock() + first, period, STD_MOVE(callback), isAsync);
}

boost::shared_ptr<TimerItem> TimerDaemon::registerHourlyTimer(
	unsigned minute, unsigned second, TimerCallback callback, bool isAsync)
{
	const AUTO(delta, getLocalTime() - (minute * 60ull + second));
	return registerTimer(MILLISECS_PER_HOUR - delta % MILLISECS_PER_HOUR, MILLISECS_PER_HOUR, STD_MOVE(callback), isAsync);
}
boost::shared_ptr<TimerItem> TimerDaemon::registerDailyTimer(
	unsigned hour, unsigned minute, unsigned second, TimerCallback callback, bool isAsync)
{
	const AUTO(delta, getLocalTime() - (hour * 3600ull + minute * 60ull + second));
	return registerTimer(MILLISECS_PER_DAY - delta % MILLISECS_PER_DAY, MILLISECS_PER_DAY, STD_MOVE(callback), isAsync);
}
boost::shared_ptr<TimerItem> TimerDaemon::registerWeeklyTimer(
	unsigned dayOfWeek, unsigned hour, unsigned minute, unsigned second, TimerCallback callback, bool isAsync)
{
	// 注意 1970-01-01 是星期四。
	const AUTO(delta, getLocalTime() - ((dayOfWeek + 3) * 86400ull + hour * 3600ull + minute * 60ull + second));
	return registerTimer(MILLISECS_PER_WEEK - delta % MILLISECS_PER_WEEK, MILLISECS_PER_WEEK, STD_MOVE(callback), isAsync);
}

void TimerDaemon::setAbsoluteTime(
	const boost::shared_ptr<TimerItem> &item, boost::uint64_t timePoint, boost::uint64_t period)
{
	PROFILE_ME;

	const Mutex::UniqueLock lock(g_mutex);
	if(period != TimerDaemon::PERIOD_NOT_MODIFIED){
		item->period = period;
	}
	g_timers.push_back(TimerQueueElement(timePoint, item, ++item->stamp));
	std::push_heap(g_timers.begin(), g_timers.end());
	g_newTimer.signal();
}
void TimerDaemon::setTime(
	const boost::shared_ptr<TimerItem> &item, boost::uint64_t first, boost::uint64_t period)
{
	return setAbsoluteTime(item, getFastMonoClock() + first, period);
}

}
