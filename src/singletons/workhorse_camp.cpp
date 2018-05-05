// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2018, LH_Mouse. All wrongs reserved.

#include "../precompiled.hpp"
#include "workhorse_camp.hpp"
#include "main_config.hpp"
#include "../thread.hpp"
#include "../mutex.hpp"
#include "../condition_variable.hpp"
#include "../atomic.hpp"
#include "../exception.hpp"
#include "../log.hpp"
#include "../raii.hpp"
#include "../promise.hpp"
#include "../profiler.hpp"
#include "../random.hpp"

namespace Poseidon {

typedef Workhorse_camp::Job_procedure Job_procedure;

namespace {
	class Workhorse_thread : NONCOPYABLE {
	private:
		struct Job_queue_element {
			boost::weak_ptr<Promise> weak_promise;
			Job_procedure procedure;
		};

	private:
		Thread m_thread;
		volatile bool m_running;

		mutable Mutex m_mutex;
		mutable Condition_variable m_new_job;
		boost::container::deque<Job_queue_element> m_queue;

	public:
		Workhorse_thread()
			: m_running(false)
			, m_queue()
		{
			//
		}

	private:
		bool pump_one_job() NOEXCEPT {
			PROFILE_ME;

			Job_queue_element *elem;
			{
				const Mutex::Unique_lock lock(m_mutex);
				if(m_queue.empty()){
					return false;
				}
				elem = &m_queue.front();
			}

			STD_EXCEPTION_PTR except;
			try {
				elem->procedure();
			} catch(std::exception &e){
				LOG_POSEIDON_WARNING("std::exception thrown: what = ", e.what());
				except = STD_CURRENT_EXCEPTION();
			} catch(...){
				LOG_POSEIDON_WARNING("Unknown exception thrown");
				except = STD_CURRENT_EXCEPTION();
			}
			const AUTO(promise, elem->weak_promise.lock());
			if(promise){
				if(except){
					promise->set_exception(STD_MOVE(except), false);
				} else {
					promise->set_success(false);
				}
			}
			const Mutex::Unique_lock lock(m_mutex);
			m_queue.pop_front();
			return true;
		}

		void thread_proc(){
			PROFILE_ME;
			LOG_POSEIDON(Logger::special_major | Logger::level_info, "Workhorse thread started.");

			unsigned timeout = 0;
			for(;;){
				bool busy;
				do {
					busy = pump_one_job();
					timeout = std::min<unsigned>(timeout * 2u + 1u, !busy * 100u);
				} while(busy);

				Mutex::Unique_lock lock(m_mutex);
				if(m_queue.empty() && !atomic_load(m_running, memory_order_consume)){
					break;
				}
				m_new_job.timed_wait(lock, timeout);
			}

			LOG_POSEIDON(Logger::special_major | Logger::level_info, "Workhorse thread stopped.");
		}

	public:
		void start(){
			const Mutex::Unique_lock lock(m_mutex);
			Thread(boost::bind(&Workhorse_thread::thread_proc, this), sslit("  W "), sslit("Workhorse")).swap(m_thread);
			atomic_store(m_running, true, memory_order_release);
		}
		void stop(){
			atomic_store(m_running, false, memory_order_release);
		}
		void safe_join(){
			wait_till_idle();

			if(m_thread.joinable()){
				m_thread.join();
			}
		}

		void wait_till_idle(){
			for(;;){
				std::size_t pending_objects;
				{
					const Mutex::Unique_lock lock(m_mutex);
					pending_objects = m_queue.size();
					if(pending_objects == 0){
						break;
					}
					m_new_job.signal();
				}
				LOG_POSEIDON(Logger::special_major | Logger::level_info, "Waiting for jobs to complete: pending_objects = ", pending_objects);

				::timespec req;
				req.tv_sec = 0;
				req.tv_nsec = 500 * 1000 * 1000;
				::nanosleep(&req, NULLPTR);
			}
		}

		std::size_t get_queue_size() const {
			const Mutex::Unique_lock lock(m_mutex);
			return m_queue.size();
		}
		void add_job(const boost::shared_ptr<Promise> &promise, Job_procedure procedure){
			PROFILE_ME;

			const Mutex::Unique_lock lock(m_mutex);
			DEBUG_THROW_UNLESS(atomic_load(m_running, memory_order_consume), Exception, sslit("Workhorse thread is being shut down"));
			Job_queue_element elem = { promise, STD_MOVE_IDN(procedure) };
			m_queue.push_back(STD_MOVE(elem));
			m_new_job.signal();
		}
	};

	volatile bool g_running = false;

	Mutex g_router_mutex;
	boost::container::vector<boost::shared_ptr<Workhorse_thread> > g_threads;

	void add_job_using_seed(const boost::shared_ptr<Promise> &promise, Job_procedure procedure, boost::uint64_t seed){
		PROFILE_ME;
		DEBUG_THROW_UNLESS(!g_threads.empty(), Basic_exception, sslit("Workhorse support is not enabled"));

		boost::shared_ptr<Workhorse_thread> thread;
		{
			const Mutex::Unique_lock lock(g_router_mutex);
			std::size_t i = static_cast<std::size_t>(seed % g_threads.size());
			thread = g_threads.at(i);
			if(!thread){
				LOG_POSEIDON(Logger::special_major | Logger::level_debug, "Creating new workhorse thread ", i);
				thread = boost::make_shared<Workhorse_thread>();
				thread->start();
				g_threads.at(i) = thread;
			}
		}
		assert(thread);
		thread->add_job(promise, STD_MOVE_IDN(procedure));
	}
}

void Workhorse_camp::start(){
	if(atomic_exchange(g_running, true, memory_order_acq_rel) != false){
		LOG_POSEIDON_FATAL("Only one daemon is allowed at the same time.");
		std::terminate();
	}
	LOG_POSEIDON(Logger::special_major | Logger::level_info, "Starting workhorse daemon...");

	const AUTO(max_thread_count, Main_config::get<std::size_t>("workhorse_max_thread_count"));
	if(max_thread_count == 0){
		LOG_POSEIDON_FATAL("You shall not set `workhorse_max_thread_count` in `main.conf` to zero.");
		std::terminate();
	}
	g_threads.resize(max_thread_count);

	LOG_POSEIDON(Logger::special_major | Logger::level_info, "Workhorse daemon started.");
}
void Workhorse_camp::stop(){
	if(atomic_exchange(g_running, false, memory_order_acq_rel) == false){
		return;
	}
	LOG_POSEIDON(Logger::special_major | Logger::level_info, "Stopping workhorse daemon...");

	for(std::size_t i = 0; i < g_threads.size(); ++i){
		const AUTO_REF(thread, g_threads.at(i));
		if(!thread){
			continue;
		}
		LOG_POSEIDON(Logger::special_major | Logger::level_info, "Stopping workhorse thread ", i);
		thread->stop();
	}
	for(std::size_t i = 0; i < g_threads.size(); ++i){
		const AUTO_REF(thread, g_threads.at(i));
		if(!thread){
			continue;
		}
		LOG_POSEIDON(Logger::special_major | Logger::level_info, "Waiting for workhorse thread ", i, " to terminate...");
		thread->safe_join();
	}

	LOG_POSEIDON(Logger::special_major | Logger::level_info, "Workhorse daemon stopped.");

	const Mutex::Unique_lock lock(g_router_mutex);
	g_threads.clear();
}

void Workhorse_camp::enqueue_isolated(const boost::shared_ptr<Promise> &promise, Job_procedure procedure){
	add_job_using_seed(promise, STD_MOVE_IDN(procedure), random_uint32());
}
void Workhorse_camp::enqueue(const boost::shared_ptr<Promise> &promise, Job_procedure procedure, std::size_t thread_hint){
	add_job_using_seed(promise, STD_MOVE_IDN(procedure), static_cast<boost::uint64_t>(thread_hint) * 134775813 / 65539);
}

}
