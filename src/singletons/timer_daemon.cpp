// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2017, LH_Mouse. All wrongs reserved.

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
#include "../checked_arithmetic.hpp"

namespace Poseidon {

typedef TimerDaemon::TimerCallback TimerCallback;

class Timer : NONCOPYABLE {
private:
	boost::uint64_t m_period;
	unsigned long m_stamp;
	TimerCallback m_callback;
	bool m_low_level;

public:
	Timer(boost::uint64_t period, TimerCallback callback, bool low_level)
		: m_period(period), m_stamp(0), m_callback(STD_MOVE_IDN(callback)), m_low_level(low_level)
	{ }

public:
	boost::uint64_t get_period() const {
		return m_period;
	}
	unsigned long get_stamp() const {
		return m_stamp;
	}
	const TimerCallback &get_callback() const {
		return m_callback;
	}
	bool is_low_level() const {
		return m_low_level;
	}

	unsigned long set_period(boost::uint64_t period){
		if(period != TimerDaemon::PERIOD_NOT_MODIFIED){
			m_period = period;
		}
		return ++m_stamp;
	}
};

namespace {
	CONSTEXPR const boost::uint64_t MS_PER_HOUR = 3600000;
	CONSTEXPR const boost::uint64_t MS_PER_DAY  = MS_PER_HOUR * 24;
	CONSTEXPR const boost::uint64_t MS_PER_WEEK = MS_PER_DAY * 7;

	class TimerJob : public JobBase {
	private:
		const boost::weak_ptr<Timer> m_weak_timer;
		const boost::uint64_t m_now;
		const boost::uint64_t m_period;

	public:
		TimerJob(boost::weak_ptr<Timer> weak_timer, boost::uint64_t now, boost::uint64_t period)
			: m_weak_timer(STD_MOVE(weak_timer)), m_now(now), m_period(period)
		{ }

	public:
		boost::weak_ptr<const void> get_category() const OVERRIDE {
			return m_weak_timer;
		}
		void perform() OVERRIDE {
			PROFILE_ME;

			const AUTO(timer, m_weak_timer.lock());
			if(!timer){
				return;
			}
			timer->get_callback()(timer, m_now, m_period);
		}
	};

	struct TimerQueueElement {
		boost::weak_ptr<Timer> timer;
		boost::uint64_t next;
		unsigned long stamp;
	};

	bool operator<(const TimerQueueElement &lhs, const TimerQueueElement &rhs){
		return lhs.next > rhs.next;
	}

	volatile bool g_running = false;
	Thread g_thread;

	Mutex g_mutex;
	ConditionVariable g_new_timer;
	boost::container::vector<TimerQueueElement> g_timers;

	bool pump_one_element() NOEXCEPT {
		PROFILE_ME;

		const AUTO(now, get_fast_mono_clock());

		boost::shared_ptr<Timer> timer;
		boost::uint64_t period;
		{
			const Mutex::UniqueLock lock(g_mutex);
		_pick_next:
			if(g_timers.empty()){
				return false;
			}
			if(now < g_timers.front().next){
				return false;
			}
			std::pop_heap(g_timers.begin(), g_timers.end());
			timer = g_timers.back().timer.lock();
			if(!timer || (timer->get_stamp() != g_timers.back().stamp)){
				g_timers.pop_back();
				goto _pick_next;
			}
			period = timer->get_period();
			if(period == 0){
				g_timers.pop_back();
			} else {
				g_timers.back().next = saturated_add(g_timers.back().next, period);
				std::push_heap(g_timers.begin(), g_timers.end());
			}
		}
		try {
			if(timer->is_low_level()){
				LOG_POSEIDON_TRACE("Dispatching low level timer: timer = ", timer);
				timer->get_callback()(timer, now, timer->get_period());
			} else {
				LOG_POSEIDON_TRACE("Preparing a timer job for dispatching: timer = ", timer);
				JobDispatcher::enqueue(boost::make_shared<TimerJob>(timer, now, period), VAL_INIT);
			}
		} catch(std::exception &e){
			LOG_POSEIDON_WARNING("std::exception thrown while dispatching timer job, what = ", e.what());
		} catch(...){
			LOG_POSEIDON_WARNING("Unknown exception thrown while dispatching timer job.");
		}
		return true;
	}

	void thread_proc(){
		PROFILE_ME;
		LOG_POSEIDON_INFO("Timer daemon started.");

		unsigned timeout = 0;
		for(;;){
			bool busy;
			do {
				busy = pump_one_element();
				timeout = std::min(timeout * 2u + 1u, !busy * 100u);
			} while(busy);

			Mutex::UniqueLock lock(g_mutex);
			if(!atomic_load(g_running, ATOMIC_CONSUME)){
				break;
			}
			g_new_timer.timed_wait(lock, timeout);
		}

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

boost::shared_ptr<Timer> TimerDaemon::register_absolute_timer(boost::uint64_t first, boost::uint64_t period, TimerCallback callback){
	PROFILE_ME;

	AUTO(timer, boost::make_shared<Timer>(period, STD_MOVE_IDN(callback), false));
	{
		const Mutex::UniqueLock lock(g_mutex);
		TimerQueueElement elem = { timer, first, timer->get_stamp() };
		g_timers.push_back(STD_MOVE(elem));
		std::push_heap(g_timers.begin(), g_timers.end());
		g_new_timer.signal();
	}
	LOG_POSEIDON_DEBUG("Created a timer which will be triggered ", saturated_sub(first, get_fast_mono_clock()), " microsecond(s) later and has a period of ", timer->get_period(), " microsecond(s).");
	return timer;
}
boost::shared_ptr<Timer> TimerDaemon::register_timer(boost::uint64_t delta_first, boost::uint64_t period, TimerCallback callback){
	const AUTO(now, get_fast_mono_clock());
	return register_absolute_timer(saturated_add(now, delta_first), period, STD_MOVE(callback));
}

boost::shared_ptr<Timer> TimerDaemon::register_hourly_timer(unsigned minute, unsigned second, TimerCallback callback, bool utc){
	const AUTO(virt_now, utc ? get_utc_time() : get_local_time());
	const AUTO(delta, checked_sub(virt_now, (minute * 60ul + second) * 1000));
	return register_timer(MS_PER_HOUR - delta % MS_PER_HOUR, MS_PER_HOUR, STD_MOVE(callback));
}
boost::shared_ptr<Timer> TimerDaemon::register_daily_timer(unsigned hour, unsigned minute, unsigned second, TimerCallback callback, bool utc){
	const AUTO(virt_now, utc ? get_utc_time() : get_local_time());
	const AUTO(delta, checked_sub(virt_now, (hour * 3600ul + minute * 60ul + second) * 1000));
	return register_timer(MS_PER_DAY - delta % MS_PER_DAY, MS_PER_DAY, STD_MOVE(callback));
}
boost::shared_ptr<Timer> TimerDaemon::register_weekly_timer(unsigned day_of_week, unsigned hour, unsigned minute, unsigned second, TimerCallback callback, bool utc){
	// 注意 1970-01-01 是星期四。
	const AUTO(virt_now, utc ? get_utc_time() : get_local_time());
	const AUTO(delta, checked_sub(virt_now, ((day_of_week + 3) * 86400ul + hour * 3600ul + minute * 60ul + second) * 1000ul));
	return register_timer(MS_PER_WEEK - delta % MS_PER_WEEK, MS_PER_WEEK, STD_MOVE(callback));
}

boost::shared_ptr<Timer> TimerDaemon::register_low_level_absolute_timer(boost::uint64_t first, boost::uint64_t period, TimerCallback callback){
	PROFILE_ME;

	AUTO(timer, boost::make_shared<Timer>(period, STD_MOVE_IDN(callback), true));
	{
		const Mutex::UniqueLock lock(g_mutex);
		TimerQueueElement elem = { timer, first, timer->get_stamp() };
		g_timers.push_back(STD_MOVE(elem));
		std::push_heap(g_timers.begin(), g_timers.end());
		g_new_timer.signal();
	}
	LOG_POSEIDON_DEBUG("Created a low level timer which will be triggered ", saturated_sub(first, get_fast_mono_clock()), " microsecond(s) later and has a period of ", timer->get_period(), " microsecond(s).");
	return timer;
}
boost::shared_ptr<Timer> TimerDaemon::register_low_level_timer(boost::uint64_t delta_first, boost::uint64_t period, TimerCallback callback){
	const AUTO(now, get_fast_mono_clock());
	return register_low_level_absolute_timer(saturated_add(now, delta_first), period, STD_MOVE(callback));
}

void TimerDaemon::set_absolute_time(const boost::shared_ptr<Timer> &timer, boost::uint64_t first, boost::uint64_t period){
	PROFILE_ME;

	const Mutex::UniqueLock lock(g_mutex);
	g_timers.emplace_back(); // This may throw std::bad_alloc.
	TimerQueueElement elem = { timer, first, timer->set_period(period) }; // This throws no exception.
	g_timers.back() = STD_MOVE_IDN(elem); // This does not throw an exception, either.
	std::push_heap(g_timers.begin(), g_timers.end());
	g_new_timer.signal();
}
void TimerDaemon::set_time(const boost::shared_ptr<Timer> &timer, boost::uint64_t delta_first, boost::uint64_t period){
	const AUTO(now, get_fast_mono_clock());
	return set_absolute_time(timer, saturated_add(now, delta_first), period);
}

}
