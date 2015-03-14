// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2015, LH_Mouse. All wrongs reserved.

#include "../precompiled.hpp"
#include "job_dispatcher.hpp"
#include <boost/thread/mutex.hpp>
#include "main_config.hpp"
#include "../job_base.hpp"
#include "../atomic.hpp"
#include "../exception.hpp"
#include "../log.hpp"
#include "../profiler.hpp"
#include "../utilities.hpp"
#include "../multi_index_map.hpp"

namespace Poseidon {

namespace {
	struct JobElement {
		boost::uint64_t dueTime;
		boost::weak_ptr<const void> category;

		boost::shared_ptr<const JobBase> job;

		mutable std::size_t retryCount;

		JobElement(boost::uint64_t dueTime_, boost::shared_ptr<const JobBase> job_)
			: dueTime(dueTime_)
			, job(STD_MOVE(job_))
			, retryCount(0)
		{
			category = job->getCategory();
			if(!(boost::weak_ptr<void>() < category) && !(category < boost::weak_ptr<void>())){
				category = job;
			}
		}
	};

	MULTI_INDEX_MAP(JobMap, JobElement,
		MULTI_MEMBER_INDEX(dueTime)
		MULTI_MEMBER_INDEX(category)
	);

	std::size_t g_maxRetryCount			= 5;
	boost::uint64_t g_retryInitDelay	= 1000;

	volatile bool g_running = false;

	boost::mutex g_mutex;
	JobMap g_jobMap;

	bool pumpOneJob(){
		PROFILE_ME;

		typedef JobMap::delegated_container::nth_index<0>::type::iterator JobIterator;

		const AUTO(now, getFastMonoClock());

		JobIterator jobIt;
		{
			const boost::mutex::scoped_lock lock(g_mutex);
			jobIt = g_jobMap.begin<0>();
			if(jobIt == g_jobMap.end<0>()){
				return false;
			}
			if(now < jobIt->dueTime){
				return false;
			}
		}

		boost::uint64_t newDueTime = 0;
		try {
			jobIt->job->perform();
		} catch(JobBase::TryAgainLater &){
			LOG_POSEIDON_INFO("JobBase::TryAgainLater thrown while dispatching job: retryCount = ", jobIt->retryCount);

			if(jobIt->retryCount < g_maxRetryCount){
				newDueTime = now + (g_retryInitDelay << jobIt->retryCount);
				++jobIt->retryCount;
			} else {
				LOG_POSEIDON_ERROR("Max retry count exceeded. Give up.");
			}
		} catch(std::exception &e){
			LOG_POSEIDON_ERROR("std::exception thrown in job dispatcher: what = ", e.what());
		} catch(...){
			LOG_POSEIDON_ERROR("Unknown exception thrown in job dispatcher.");
		}
		if(newDueTime != 0){
			const boost::mutex::scoped_lock lock(g_mutex);
			const AUTO(range, g_jobMap.equalRange<1>(jobIt->category));
			for(AUTO(it, range.first); it != range.second; ++it){
				g_jobMap.setKey<1, 0>(it, newDueTime);
				++newDueTime;
			}
		} else {
			const boost::mutex::scoped_lock lock(g_mutex);
			g_jobMap.erase<0>(jobIt);
		}

		return true;
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

	while(pumpOneJob()){
		// noop
	}
}

void JobDispatcher::doModal(){
	LOG_POSEIDON_INFO("Entering modal loop...");

	if(atomicExchange(g_running, true, ATOMIC_ACQ_REL) != false){
		LOG_POSEIDON_FATAL("Only one modal loop is allowed at the same time.");
		std::abort();
	}

	for(;;){
		while(pumpOneJob()){
			// noop
		}

		if(!atomicLoad(g_running, ATOMIC_ACQUIRE)){
			break;
		}
		::usleep(100000);
	}
}
void JobDispatcher::quitModal(){
	atomicStore(g_running, false, ATOMIC_RELEASE);
}

void JobDispatcher::enqueue(boost::shared_ptr<const JobBase> job, boost::uint64_t delay){
	const boost::mutex::scoped_lock lock(g_mutex);
	g_jobMap.insert(JobElement(getFastMonoClock() + delay, STD_MOVE(job)));
}

}
