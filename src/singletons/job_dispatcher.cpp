// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2015, LH_Mouse. All wrongs reserved.

#include "../precompiled.hpp"
#include "job_dispatcher.hpp"
#include "main_config.hpp"
#include "../job_base.hpp"
#include "../atomic.hpp"
#include "../exception.hpp"
#include "../log.hpp"
#include "../profiler.hpp"
#include "../mutex.hpp"
#include "../condition_variable.hpp"
#include "../time.hpp"
#include "../multi_index_map.hpp"

namespace Poseidon {

namespace {
	template<typename Tx, typename Ty>
	bool ownerEqual(const boost::shared_ptr<Tx> &lhs, const boost::shared_ptr<Ty> &rhs){
		return !lhs.owner_before(rhs) && !rhs.owner_before(lhs);
	}
	template<typename Tx, typename Ty>
	bool ownerEqual(const boost::weak_ptr<Tx> &lhs, const boost::shared_ptr<Ty> &rhs){
		return !lhs.owner_before(rhs) && !rhs.owner_before(lhs);
	}
	template<typename Tx, typename Ty>
	bool ownerEqual(const boost::shared_ptr<Tx> &lhs, const boost::weak_ptr<Ty> &rhs){
		return !lhs.owner_before(rhs) && !rhs.owner_before(lhs);
	}
	template<typename Tx, typename Ty>
	bool ownerEqual(const boost::weak_ptr<Tx> &lhs, const boost::weak_ptr<Ty> &rhs){
		return !lhs.owner_before(rhs) && !rhs.owner_before(lhs);
	}

	struct RetryCountElement {
		boost::shared_ptr<const void> ptr;
		std::size_t count;

		RetryCountElement(boost::shared_ptr<const void> ptr_, std::size_t count_)
			: ptr(STD_MOVE(ptr_)), count(count_)
		{
		}
	};

	struct RetryCountComparator {
		bool operator()(const RetryCountElement &lhs, const boost::shared_ptr<const void> &rhs) const {
			return lhs.ptr.owner_before(rhs);
		}
		bool operator()(const boost::shared_ptr<const void> &lhs, const RetryCountElement &rhs) const {
			return lhs.owner_before(rhs.ptr);
		}
	};

	struct JobElement {
		boost::shared_ptr<const JobBase> job;
		boost::shared_ptr<const bool> withdrawn;

		boost::uint64_t dueTime;
		boost::weak_ptr<const void> category;

		mutable std::vector<RetryCountElement> retryCounts;

		JobElement(boost::shared_ptr<const JobBase> job_, boost::shared_ptr<const bool> withdrawn_,
			boost::uint64_t dueTime_, boost::weak_ptr<const void> category_)
			: job(STD_MOVE(job_)), withdrawn(STD_MOVE(withdrawn_))
			, dueTime(dueTime_), category(STD_MOVE(category_))
		{
			assert(!ownerEqual(category, boost::weak_ptr<void>()));
		}
	};

	inline void swap(JobElement &lhs, JobElement &rhs) NOEXCEPT {
		using std::swap;
		swap(lhs.job, rhs.job);
		swap(lhs.withdrawn, rhs.withdrawn);
		swap(lhs.dueTime, rhs.dueTime);
		swap(lhs.category, rhs.category);
		swap(lhs.retryCounts, rhs.retryCounts);
	}

	MULTI_INDEX_MAP(JobMap, JobElement,
		MULTI_MEMBER_INDEX(dueTime)
		MULTI_MEMBER_INDEX(category)
	)

	std::size_t g_maxRetryCount			= 6;
	boost::uint64_t g_retryInitDelay	= 100;

	volatile bool g_running = false;

	Mutex g_mutex;
	ConditionVariable g_newJob;
	JobMap g_jobMap;

	bool pumpOneJob() NOEXCEPT {
		PROFILE_ME;

		typedef JobMap::delegated_container::nth_index<0>::type::iterator JobIterator;

		const AUTO(now, getFastMonoClock());

		JobIterator jobIt;
		{
			const Mutex::UniqueLock lock(g_mutex);
			jobIt = g_jobMap.begin<0>();
			if(jobIt == g_jobMap.end<0>()){
				return false;
			}
			if(atomicLoad(g_running, ATOMIC_ACQUIRE) && (now < jobIt->dueTime)){
				return false;
			}
		}

		boost::uint64_t newDueTime = 0;
		if(jobIt->withdrawn && *jobIt->withdrawn){
			LOG_POSEIDON_DEBUG("Job withdrawn");
		} else {
			try {
				try {
					jobIt->job->perform();
				} catch(JobBase::TryAgainLater &e){
					const AUTO(context, e.getContext());
					AUTO(it, std::upper_bound(jobIt->retryCounts.begin(), jobIt->retryCounts.end(), context, RetryCountComparator()));
					if((it != jobIt->retryCounts.begin()) && ownerEqual(it[-1].ptr, context)){
						--it;
					} else {
						it = jobIt->retryCounts.insert(it, RetryCountElement(context, 0));
					}
					LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_INFO,
						"JobBase::TryAgainLater thrown while dispatching job: retryCount = ", it->count);

					if(it->count >= g_maxRetryCount){
						LOG_POSEIDON_ERROR("Max retry count exceeded.");
						DEBUG_THROW(Exception, sslit("Max retry count exceeded"));
					}
					newDueTime = now + (g_retryInitDelay << it->count);
					++(it->count);
				}
			} catch(std::exception &e){
				LOG_POSEIDON_WARNING("std::exception thrown in job dispatcher: what = ", e.what());
			} catch(...){
				LOG_POSEIDON_WARNING("Unknown exception thrown in job dispatcher.");
			}
		}
		{
			const Mutex::UniqueLock lock(g_mutex);
			if(newDueTime != 0){
				const AUTO(deltaTime, newDueTime - jobIt->dueTime);
				const AUTO(range, g_jobMap.equalRange<1>(jobIt->category));
				for(AUTO(it, range.first); it != range.second; ++it){
					g_jobMap.setKey<1, 0>(it, it->dueTime + deltaTime);
				}
			} else {
				g_jobMap.erase<0>(jobIt);
			}
		}

		return true;
	}
}

void JobDispatcher::start(){
	LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_INFO, "Starting job dispatcher...");

	AUTO_REF(conf, MainConfig::get());

	conf.get(g_maxRetryCount, "job_max_retry_count");
	LOG_POSEIDON_DEBUG("Max retry count = ", g_maxRetryCount);

	conf.get(g_retryInitDelay, "job_retry_init_delay");
	LOG_POSEIDON_DEBUG("Retry init delay = ", g_retryInitDelay);
}
void JobDispatcher::stop(){
	LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_INFO, "Stopping job dispatcher...");

	pumpAll();
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

		Mutex::UniqueLock lock(g_mutex);
		g_newJob.timedWait(lock, 50);
	}
}
void JobDispatcher::quitModal(){
	atomicStore(g_running, false, ATOMIC_RELEASE);
}

void JobDispatcher::enqueue(boost::shared_ptr<const JobBase> job, boost::uint64_t delay,
	boost::shared_ptr<const bool> withdrawn)
{
	PROFILE_ME;

	AUTO(dueTime, getFastMonoClock() + delay);
	AUTO(category, job->getCategory());

	const Mutex::UniqueLock lock(g_mutex);
	{
		const AUTO(range, g_jobMap.equalRange<1>(category));
		for(AUTO(it, range.first); it != range.second; ++it){
			if(dueTime > it->dueTime){
				continue;
			}
			dueTime = it->dueTime + 1;
		}
		if(ownerEqual(category, boost::weak_ptr<void>())){
			category = job;
		}
	}
	g_jobMap.insert(JobElement(STD_MOVE(job), STD_MOVE(withdrawn), dueTime, STD_MOVE(category)));
	g_newJob.signal();
}
void JobDispatcher::pumpAll(){
	LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_INFO, "Flushing all queued jobs...");

	while(pumpOneJob()){
		// noop
	}
}

}
