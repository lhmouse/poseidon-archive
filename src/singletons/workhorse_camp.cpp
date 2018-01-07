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

typedef WorkhorseCamp::JobProcedure JobProcedure;

namespace {
	class WorkhorseThread : NONCOPYABLE {
	private:
		struct JobQueueElement {
			boost::weak_ptr<Promise> weak_promise;
			JobProcedure procedure;
		};

	private:
		Thread m_thread;
		volatile bool m_running;

		mutable Mutex m_mutex;
		mutable ConditionVariable m_new_job;
		boost::container::deque<JobQueueElement> m_queue;

	public:
		WorkhorseThread()
			: m_running(false)
		{ }

	private:
		bool pump_one_job() NOEXCEPT {
			PROFILE_ME;

			JobQueueElement *elem;
			{
				const Mutex::UniqueLock lock(m_mutex);
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
			const Mutex::UniqueLock lock(m_mutex);
			m_queue.pop_front();
			return true;
		}

		void thread_proc(){
			PROFILE_ME;
			LOG_POSEIDON_INFO("Workhorse thread started.");

			unsigned timeout = 0;
			for(;;){
				bool busy;
				do {
					busy = pump_one_job();
					timeout = std::min<unsigned>(timeout * 2u + 1u, !busy * 100u);
				} while(busy);

				Mutex::UniqueLock lock(m_mutex);
				if(m_queue.empty() && !atomic_load(m_running, ATOMIC_CONSUME)){
					break;
				}
				m_new_job.timed_wait(lock, timeout);
			}

			LOG_POSEIDON_INFO("Workhorse thread stopped.");
		}

	public:
		void start(){
			const Mutex::UniqueLock lock(m_mutex);
			Thread(boost::bind(&WorkhorseThread::thread_proc, this), "  W ").swap(m_thread);
			atomic_store(m_running, true, ATOMIC_RELEASE);
		}
		void stop(){
			atomic_store(m_running, false, ATOMIC_RELEASE);
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
					const Mutex::UniqueLock lock(m_mutex);
					pending_objects = m_queue.size();
					if(pending_objects == 0){
						break;
					}
					m_new_job.signal();
				}
				LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_INFO, "Waiting for jobs to complete: pending_objects = ", pending_objects);

				::timespec req;
				req.tv_sec = 0;
				req.tv_nsec = 500 * 1000 * 1000;
				::nanosleep(&req, NULLPTR);
			}
		}

		std::size_t get_queue_size() const {
			const Mutex::UniqueLock lock(m_mutex);
			return m_queue.size();
		}
		void add_job(const boost::shared_ptr<Promise> &promise, JobProcedure procedure){
			PROFILE_ME;

			const Mutex::UniqueLock lock(m_mutex);
			DEBUG_THROW_UNLESS(atomic_load(m_running, ATOMIC_CONSUME), Exception, sslit("Workhorse thread is being shut down"));
			JobQueueElement elem = { promise, STD_MOVE_IDN(procedure) };
			m_queue.push_back(STD_MOVE(elem));
			m_new_job.signal();
		}
	};

	volatile bool g_running = false;

	Mutex g_router_mutex;
	boost::container::vector<boost::shared_ptr<WorkhorseThread> > g_threads;

	void add_job_using_seed(const boost::shared_ptr<Promise> &promise, JobProcedure procedure, std::size_t seed){
		PROFILE_ME;
		DEBUG_THROW_UNLESS(!g_threads.empty(), BasicException, sslit("Workhorse support is not enabled"));

		boost::shared_ptr<WorkhorseThread> thread;
		{
			const Mutex::UniqueLock lock(g_router_mutex);
			std::size_t i = seed % g_threads.size();
			thread = g_threads.at(i);
			if(!thread){
				LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_DEBUG, "Creating new workhorse thread ", i);
				thread = boost::make_shared<WorkhorseThread>();
				thread->start();
				g_threads.at(i) = thread;
			}
		}
		assert(thread);
		thread->add_job(STD_MOVE(promise), STD_MOVE_IDN(procedure));
	}
}

void WorkhorseCamp::start(){
	if(atomic_exchange(g_running, true, ATOMIC_ACQ_REL) != false){
		LOG_POSEIDON_FATAL("Only one daemon is allowed at the same time.");
		std::abort();
	}
	LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_INFO, "Starting workhorse daemon...");

	const AUTO(max_thread_count, MainConfig::get<std::size_t>("workhorse_max_thread_count"));
	if(max_thread_count == 0){
		LOG_POSEIDON_FATAL("You shall not set `workhorse_max_thread_count` in `main.conf` to zero.");
		std::abort();
	}
	g_threads.resize(max_thread_count);

	LOG_POSEIDON_INFO("Workhorse daemon started.");
}
void WorkhorseCamp::stop(){
	if(atomic_exchange(g_running, false, ATOMIC_ACQ_REL) == false){
		return;
	}
	LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_INFO, "Stopping workhorse daemon...");

	for(std::size_t i = 0; i < g_threads.size(); ++i){
		const AUTO_REF(thread, g_threads.at(i));
		if(!thread){
			continue;
		}
		LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_INFO, "Stopping workhorse thread ", i);
		thread->stop();
	}
	for(std::size_t i = 0; i < g_threads.size(); ++i){
		const AUTO_REF(thread, g_threads.at(i));
		if(!thread){
			continue;
		}
		LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_INFO, "Waiting for workhorse thread ", i, " to terminate...");
		thread->safe_join();
	}
	g_threads.clear();

	LOG_POSEIDON_INFO("Workhorse daemon stopped.");
}

void WorkhorseCamp::enqueue_isolated(const boost::shared_ptr<Promise> &promise, JobProcedure procedure){
	add_job_using_seed(promise, STD_MOVE_IDN(procedure), random_uint32());
}
void WorkhorseCamp::enqueue(const boost::shared_ptr<Promise> &promise, JobProcedure procedure, std::size_t thread_hint){
	add_job_using_seed(promise, STD_MOVE_IDN(procedure), (boost::uint64_t)thread_hint * 134775813 / 65539);
}

}
