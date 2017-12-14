// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2017, LH_Mouse. All wrongs reserved.

#include "../precompiled.hpp"
#include "workhorse_camp.hpp"
#include "main_config.hpp"
#include <boost/container/flat_map.hpp>
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
			boost::shared_ptr<Promise> promise;
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
#ifdef POSEIDON_CXX11
			std::exception_ptr except;
#else
			boost::exception_ptr except;
#endif
			try {
				elem->procedure();
			} catch(std::exception &e){
				LOG_POSEIDON_WARNING("std::exception thrown: what = ", e.what());
#ifdef POSEIDON_CXX11
				except = std::current_exception();
#else
				except = boost::copy_exception(std::runtime_error(e.what()));
#endif
			} catch(...){
				LOG_POSEIDON_WARNING("Unknown exception thrown");
#ifdef POSEIDON_CXX11
				except = std::current_exception();
#else
				except = boost::copy_exception(std::bad_exception());
#endif
			}
			if(!elem->promise->is_satisfied()){
				try {
					if(!except){
						elem->promise->set_success();
					} else {
						elem->promise->set_exception(except);
					}
				} catch(std::exception &e){
					LOG_POSEIDON_ERROR("std::exception thrown: what = ", e.what());
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
				LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_INFO,
					"Waiting for jobs to complete: pending_objects = ", pending_objects);

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
		void add_job(boost::shared_ptr<Promise> promise, JobProcedure procedure){
			PROFILE_ME;

			const Mutex::UniqueLock lock(m_mutex);
			if(!atomic_load(m_running, ATOMIC_CONSUME)){
				LOG_POSEIDON_ERROR("Workhorse thread is being shut down.");
				DEBUG_THROW(Exception, sslit("Workhorse thread is being shut down"));
			}
			JobQueueElement elem = { STD_MOVE(promise), STD_MOVE_IDN(procedure) };
			m_queue.push_back(STD_MOVE(elem));
			m_new_job.signal();
		}
	};

	volatile bool g_running = false;

	Poseidon::Mutex g_router_mutex;
	std::vector<boost::shared_ptr<WorkhorseThread> > g_threads;

	void submit_job_using_seed(boost::shared_ptr<Promise> promise, JobProcedure procedure, std::size_t seed){
		PROFILE_ME;

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

	const AUTO(max_thread_count, MainConfig::get<std::size_t>("workhorse_max_thread_count", 1));
	g_threads.resize(std::max<std::size_t>(max_thread_count, 1));

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

	LOG_POSEIDON_INFO("workhorse daemon stopped.");
}

void WorkhorseCamp::enqueue_isolated(boost::shared_ptr<Promise> promise, JobProcedure procedure){
	DEBUG_THROW_ASSERT(!g_threads.empty());

	std::size_t seed = random_uint32();
	submit_job_using_seed(STD_MOVE(promise), STD_MOVE_IDN(procedure), seed);
}
void WorkhorseCamp::enqueue(boost::shared_ptr<Promise> promise, JobProcedure procedure, std::size_t thread_hint){
	DEBUG_THROW_ASSERT(!g_threads.empty());

	std::size_t seed = static_cast<boost::uint64_t>(thread_hint) * 134775813 / 65539;
	submit_job_using_seed(STD_MOVE(promise), STD_MOVE_IDN(procedure), seed);
}

}
