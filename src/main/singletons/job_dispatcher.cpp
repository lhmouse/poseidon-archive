// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2015, LH_Mouse. All wrongs reserved.

#include "../precompiled.hpp"
#include "job_dispatcher.hpp"
#include <boost/thread/mutex.hpp>
#include <boost/thread/condition_variable.hpp>
#include "main_config.hpp"
#include "../job_base.hpp"
#include "../atomic.hpp"
#include "../exception.hpp"
#include "../log.hpp"
#include "../profiler.hpp"
#include "../utilities.hpp"

namespace Poseidon {

namespace {
	const boost::weak_ptr<const void> NULL_WEAK_PTR;

	std::size_t g_maxRetryCount			= 5;
	boost::uint64_t g_retryInitDelay	= 100;

	volatile bool g_running = false;

	boost::mutex g_mutex;
	std::deque<boost::shared_ptr<JobBase> > g_queue;
	boost::condition_variable g_newJobAvail;

	struct SuspendedQueue {
		std::deque<boost::shared_ptr<JobBase> > queue;

		// 只记录第一个。
		boost::uint64_t until;
		std::size_t retryCount;
	};

	std::map<boost::weak_ptr<const void>, SuspendedQueue> g_suspendedQueues;

	bool flushAllJobs(){
		PROFILE_ME;

		bool ret = false;

		std::deque<boost::shared_ptr<JobBase> > queue;
		{
			boost::mutex::scoped_lock lock(g_mutex);
			queue.swap(g_queue);
		}

		bool busy;
		do {
			const AUTO(now, getFastMonoClock());
			busy = false;

			if(!queue.empty()){
				// 处理主队列中的任务。

				try {
					const AUTO(job, queue.front());

					AUTO(category, job->getCategory());
					if(!(NULL_WEAK_PTR < category) && !(category < NULL_WEAK_PTR)){
						category = job;
					}

					AUTO(it, g_suspendedQueues.find(category));
					if(it == g_suspendedQueues.end()){
						// 只有在之前不存在同一类别的任务被推迟的情况下才能执行。
						try {
							job->perform();
						} catch(JobBase::TryAgainLater &){
							LOG_POSEIDON_INFO("JobBase::TryAgainLater thrown while dispatching job. Suspend it.");

							if(g_maxRetryCount == 0){
								DEBUG_THROW(Exception, SharedNts::observe("Max retry count exceeded"));
							}
							it = g_suspendedQueues.insert(std::make_pair(category, SuspendedQueue())).first;
						}
					}
					if(it != g_suspendedQueues.end()){
						// 不相等说明我们需要将当前任务放进队列。
						it->second.queue.push_back(job);
						if(it->second.queue.size() == 1){
							// 第一个被推迟的任务，设定初始时间。
							it->second.until = now + g_retryInitDelay;
							it->second.retryCount = 1;
						}
					}
				} catch(std::exception &e){
					LOG_POSEIDON_ERROR("std::exception thrown in job dispatcher: what = ", e.what());
				} catch(...){
					LOG_POSEIDON_ERROR("Unknown exception thrown in job dispatcher.");
				}
				queue.pop_front();

				ret = true;
				busy = true;
			}

			AUTO(next, g_suspendedQueues.begin());
			while(next != g_suspendedQueues.end()){
				// 处理延迟队列中的任务。

				const AUTO(it, next);
				++next;
				if(it->second.queue.empty()){
					g_suspendedQueues.erase(it);
					continue;
				}
				if(now < it->second.until){
					continue;
				}
				try {
					const AUTO(job, it->second.queue.front());

					try {
						job->perform();
					} catch(JobBase::TryAgainLater &){
						LOG_POSEIDON_INFO("JobBase::TryAgainLater thrown while dispatching suspended job.");

						if(it->second.retryCount >= g_maxRetryCount){
							DEBUG_THROW(Exception, SharedNts::observe("Max retry count exceeded"));
						}
						it->second.until = now + (g_retryInitDelay << it->second.retryCount);
						++(it->second.retryCount);
						goto _dontErase;
					}
				} catch(std::exception &e){
					LOG_POSEIDON_ERROR("std::exception thrown in job dispatcher: what = ", e.what());
				} catch(...){
					LOG_POSEIDON_ERROR("Unknown exception thrown in job dispatcher.");
				}
				it->second.queue.pop_front();
				it->second.until = 0;
				it->second.retryCount = 0;
			_dontErase:
				;

				ret = true;
				busy = true;
			}
		} while(busy);

		return ret;
	}
}

void JobDispatcher::start(){
	LOG_POSEIDON_INFO("Starting job dispatcher...");

	AUTO_REF(conf, MainConfig::getConfigFile());

	conf.get(g_maxRetryCount, "job_max_retry_count");
	LOG_POSEIDON_DEBUG("Max retry count = ", g_maxRetryCount);

	conf.get(g_retryInitDelay, "job_retry_init_delay");
	LOG_POSEIDON_DEBUG("Retry init delay = ", g_retryInitDelay);
}
void JobDispatcher::stop(){
	LOG_POSEIDON_INFO("Flushing all queued jobs...");

	for(;;){
		if(!flushAllJobs()){
			break;
		}
	}
}

void JobDispatcher::doModal(){
	LOG_POSEIDON_INFO("Entering modal loop...");

	if(atomicExchange(g_running, true, ATOMIC_ACQ_REL) != false){
		LOG_POSEIDON_FATAL("Only one modal loop is allowed at the same time.");
		std::abort();
	}

	for(;;){
		if(!flushAllJobs()){
			boost::mutex::scoped_lock lock(g_mutex);
			if(!atomicLoad(g_running, ATOMIC_ACQUIRE)){
				break;
			}
			g_newJobAvail.timed_wait(lock, boost::posix_time::milliseconds(100));
		}
	}
}
void JobDispatcher::quitModal(){
	if(atomicExchange(g_running, false, ATOMIC_ACQ_REL) == false){
		return;
	}
	{
		const boost::mutex::scoped_lock lock(g_mutex);
		g_newJobAvail.notify_all();
	}
}

void JobDispatcher::enqueue(boost::shared_ptr<JobBase> job){
	const boost::mutex::scoped_lock lock(g_mutex);
	g_queue.push_back(STD_MOVE(job));
	g_newJobAvail.notify_all();
}

}
