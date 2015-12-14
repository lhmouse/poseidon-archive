// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2015, LH_Mouse. All wrongs reserved.

#include "../precompiled.hpp"
#include "timer_daemon.hpp"
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
	bool is_async;
	unsigned long stamp;

	TimerItem(boost::uint64_t period_, boost::shared_ptr<const TimerCallback> callback_, bool is_async_)
		: period(period_), callback(STD_MOVE(callback_)), is_async(is_async_), stamp(0)
	{
		LOG_POSEIDON_DEBUG("Created timer: period = ", period, ", is_async = ", is_async);
	}
	~TimerItem(){
		LOG_POSEIDON_DEBUG("Destroyed timer: period = ", period, ", is_async = ", is_async);
	}
};

namespace {
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
		boost::weak_ptr<const void> get_category() const OVERRIDE {
			return m_item;
		}
		void perform() OVERRIDE {
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
	ConditionVariable g_new_timer;
	std::vector<TimerQueueElement> g_timers;

	bool pump_one_element() NOEXCEPT {
		PROFILE_ME;

		const boost::uint64_t now = get_fast_mono_clock();

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
			if(item->is_async){
				LOG_POSEIDON_TRACE("Dispatching async timer");
				(*item->callback)(item, now, item->period);
			} else {
				LOG_POSEIDON_TRACE("Preparing a timer job for dispatching");
				enqueue_job(boost::make_shared<TimerJob>(item, now));
			}
		} catch(std::exception &e){
			LOG_POSEIDON_WARNING("std::exception thrown while dispatching timer job, what = ", e.what());
		} catch(...){
			LOG_POSEIDON_WARNING("Unknown exception thrown while dispatching timer job.");
		}

		return true;
	}

	void daemon_loop(){
		PROFILE_ME;

		for(;;){
			while(pump_one_element()){
				// noop
			}

			if(!atomic_load(g_running, ATOMIC_CONSUME)){
				break;
			}

			Mutex::UniqueLock lock(g_mutex);
			g_new_timer.timed_wait(lock, 100);
		}
	}

	void thread_proc(){
		PROFILE_ME;
		LOG_POSEIDON_INFO("Timer daemon started.");

		daemon_loop();

		LOG_POSEIDON_INFO("Timer daemon stopped.");
	}
}

void TimerDaemon::start(){
	if(atomic_exchange(g_running, true, ATOMIC_ACQ_REL) != false){
		LOG_POSEIDON_FATAL("Only one daemon is allowed at the same time.");
		std::abort();
	}
	LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_INFO, "Starting timer daemon...");

	Thread(thread_proc, "  T ").swap(g_thread);
}
void TimerDaemon::stop(){
	if(atomic_exchange(g_running, false, ATOMIC_ACQ_REL) == false){
		return;
	}
	LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_INFO, "Stopping timer daemon...");

	if(g_thread.joinable()){
		g_thread.join();
	}
	g_timers.clear();
}

boost::shared_ptr<TimerItem> TimerDaemon::register_absolute_timer(
	boost::uint64_t time_point, boost::uint64_t period, TimerCallback callback, bool is_async)
{
	PROFILE_ME;

	AUTO(item, boost::make_shared<TimerItem>(period, boost::make_shared<TimerCallback>(STD_MOVE_IDN(callback)), is_async));
	{
		const Mutex::UniqueLock lock(g_mutex);
		g_timers.push_back(TimerQueueElement(time_point, item, item->stamp));
		std::push_heap(g_timers.begin(), g_timers.end());
		g_new_timer.signal();
	}
	LOG_POSEIDON_DEBUG("Created a(n) ", is_async ? "async " : "", "timer item which will be triggered ",
		__extension__({ const AUTO(now, get_fast_mono_clock()); (time_point < now) ? 0 : (time_point - now); }),
		" microsecond(s) later and has a period of ", item->period, " microsecond(s).");
	return item;
}
boost::shared_ptr<TimerItem> TimerDaemon::register_timer(
	boost::uint64_t first, boost::uint64_t period, TimerCallback callback, bool is_async)
{
	return register_absolute_timer(get_fast_mono_clock() + first, period, STD_MOVE(callback), is_async);
}

boost::shared_ptr<TimerItem> TimerDaemon::register_hourly_timer(
	unsigned minute, unsigned second, TimerCallback callback, bool is_async)
{
	const boost::uint64_t MS_PER_HOUR = 3600 * 1000;
	const AUTO(delta, get_local_time() - (minute * 60ull + second) * 1000);
	return register_timer(MS_PER_HOUR - delta % MS_PER_HOUR, MS_PER_HOUR, STD_MOVE(callback), is_async);
}
boost::shared_ptr<TimerItem> TimerDaemon::register_daily_timer(
	unsigned hour, unsigned minute, unsigned second, TimerCallback callback, bool is_async)
{
	const boost::uint64_t MS_PER_DAY = 24 * 3600 * 1000;
	const AUTO(delta, get_local_time() - (hour * 3600ull + minute * 60ull + second) * 1000);
	return register_timer(MS_PER_DAY - delta % MS_PER_DAY, MS_PER_DAY, STD_MOVE(callback), is_async);
}
boost::shared_ptr<TimerItem> TimerDaemon::register_weekly_timer(
	unsigned day_of_week, unsigned hour, unsigned minute, unsigned second, TimerCallback callback, bool is_async)
{
	const boost::uint64_t MS_PER_WEEK = 7 * 24 * 3600 * 1000;
	// 注意 1970-01-01 是星期四。
	const AUTO(delta, get_local_time() - ((day_of_week + 3) * 86400ull + hour * 3600ull + minute * 60ull + second) * 1000);
	return register_timer(MS_PER_WEEK - delta % MS_PER_WEEK, MS_PER_WEEK, STD_MOVE(callback), is_async);
}

void TimerDaemon::set_absolute_time(const boost::shared_ptr<TimerItem> &item, boost::uint64_t time_point, boost::uint64_t period){
	PROFILE_ME;

	const Mutex::UniqueLock lock(g_mutex);
	if(period != TimerDaemon::PERIOD_NOT_MODIFIED){
		item->period = period;
	}
	g_timers.push_back(TimerQueueElement(time_point, item, ++item->stamp));
	std::push_heap(g_timers.begin(), g_timers.end());
	g_new_timer.signal();
}
void TimerDaemon::set_time(const boost::shared_ptr<TimerItem> &item, boost::uint64_t first, boost::uint64_t period){
	return set_absolute_time(item, get_fast_mono_clock() + first, period);
}

}
