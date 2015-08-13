// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2015, LH_Mouse. All wrongs reserved.

#include "../precompiled.hpp"
#include "job_dispatcher.hpp"
#include "main_config.hpp"
#include <csetjmp>
#include "../job_base.hpp"
#include "../atomic.hpp"
#include "../exception.hpp"
#include "../log.hpp"
#include "../profiler.hpp"
#include "../mutex.hpp"
#include "../condition_variable.hpp"
#include "../time.hpp"
#include "../raii.hpp"

namespace Poseidon {

namespace {
	enum FiberState {
		FS_UNUSED		= 0,
		FS_READY		= 1,
		FS_RUNNING		= 2,
		FS_YIELDED		= 3,
	};

	struct JobElement {
		boost::shared_ptr<const JobBase> job;
		boost::function<bool ()> pred;
		boost::shared_ptr<const bool> withdrawn;

		JobElement(boost::shared_ptr<const JobBase> job_,
			boost::function<bool ()> pred_, boost::shared_ptr<const bool> withdrawn_)
			: job(STD_MOVE(job_)), pred(STD_MOVE_IDN(pred_)), withdrawn(STD_MOVE(withdrawn_))
		{
		}
	};

	struct FiberControl {
		std::deque<JobElement> queue;

		FiberState state;
		std::jmp_buf outer;
		std::jmp_buf inner;
		char stack[0x100000]; // 1MiB

		FiberControl(){
		}
	};

	FiberControl *g_currentFiber = NULLPTR;

	__attribute__((__noinline__))
	void scheduleFiber(FiberControl *param) NOEXCEPT {
		PROFILE_ME;

		assert(!param->queue.empty());

		if((param->state != FS_READY) && (param->state != FS_YIELDED)){
			LOG_POSEIDON_FATAL("Fiber can't be scheduled: state = ", static_cast<int>(param->state));
			std::abort();
		}

		AUTO_REF(fiber, g_currentFiber);
		fiber = param;

		if(setjmp(fiber->outer) == 0){
			if(fiber->state == FS_READY){
				__asm__(
#ifdef __x86_64__
					"movq %%rax, %%rsp \n"
#else
					"movl %%eax, %%esp \n"
#endif
					: : "a"(END(fiber->stack)) :
				);

				fiber->state = FS_RUNNING;
				LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_DEBUG, "Entering fiber ", static_cast<void *>(fiber));
				try {
					fiber->queue.front().job->perform();
				} catch(std::exception &e){
					LOG_POSEIDON_WARNING("std::exception thrown: what = ", e.what());
				} catch(...){
					LOG_POSEIDON_WARNING("Unknown exception thrown");
				}
				LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_DEBUG, "Exited from fiber ", static_cast<void *>(fiber));

				fiber->state = FS_READY;
				std::longjmp(fiber->outer, 1);
			} else {
				fiber->state = FS_RUNNING;
				std::longjmp(fiber->inner, 1);
			}
			std::abort();
		}

		g_currentFiber = NULLPTR;
	}

	volatile bool g_running = false;

	Mutex g_mutex;
	ConditionVariable g_newJob;

	std::list<FiberControl> g_fiberPool;
	std::map<boost::weak_ptr<const void>, std::list<FiberControl> > g_fiberMap;

	void reallyPumpJobs() NOEXCEPT {
		PROFILE_ME;

		bool busy;
		do {
			busy = false;

			Mutex::UniqueLock lock(g_mutex);
			AUTO(it, g_fiberMap.begin());
			while(it != g_fiberMap.end()){
				if(it->second.empty()){
					g_fiberMap.erase(it++);
					continue;
				}
				if(it->second.front().queue.empty()){
					g_fiberPool.splice(g_fiberPool.end(), it->second);
					g_fiberMap.erase(it++);
					continue;
				}
				lock.unlock();

				AUTO_REF(elem, it->second.front().queue.front());
				bool done;

				if(elem.withdrawn && *elem.withdrawn){
					LOG_POSEIDON_DEBUG("Job is withdrawn");
					done = true;
				} else if(elem.pred && !elem.pred()){
					LOG_POSEIDON_TRACE("Job is not ready to be scheduled");
					done = false;
				} else {
					boost::function<bool ()>().swap(elem.pred);

					scheduleFiber(&it->second.front());
					done = (it->second.front().state == FS_READY);
				}

				lock.lock();
				if(done){
					it->second.front().queue.pop_front();
				}
				++it;
			}
		} while(busy);
	}
}

void JobDispatcher::start(){
	LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_INFO, "Starting job dispatcher...");

//	MainConfig::get(g_maxRetryCount, "job_max_retry_count");
//	LOG_POSEIDON_DEBUG("Max retry count = ", g_maxRetryCount);
}
void JobDispatcher::stop(){
	LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_INFO, "Stopping job dispatcher...");

	AUTO(lastInfoTime, getFastMonoClock());
	for(;;){
		std::size_t pendingJobs;
		{
			const Mutex::UniqueLock lock(g_mutex);
			pendingJobs = g_fiberMap.size();
		}
		if(pendingJobs == 0){
			break;
		}

		const AUTO(now, getFastMonoClock());
		if(lastInfoTime + 500 < now){
			LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_INFO, "There are ", pendingJobs, " job queue(s) remaining.");
			lastInfoTime = now;
		}

		reallyPumpJobs();
	}
}

void JobDispatcher::doModal(){
	LOG_POSEIDON_INFO("Entering modal loop...");

	if(atomicExchange(g_running, true, ATOMIC_ACQ_REL) != false){
		LOG_POSEIDON_FATAL("Only one modal loop is allowed at the same time.");
		std::abort();
	}

	for(;;){
		reallyPumpJobs();

		Mutex::UniqueLock lock(g_mutex);
		if(!atomicLoad(g_running, ATOMIC_CONSUME) && g_fiberMap.empty()){
			break;
		}
		g_newJob.timedWait(lock, 100);
	}
}
void JobDispatcher::quitModal(){
	atomicStore(g_running, false, ATOMIC_RELEASE);
}

void JobDispatcher::enqueue(boost::shared_ptr<const JobBase> job,
	boost::function<bool ()> pred, boost::shared_ptr<const bool> withdrawn)
{
	PROFILE_ME;

	const boost::weak_ptr<const void> nullWeakPtr;
	AUTO(category, job->getCategory());
	if(!(category < nullWeakPtr) && !(nullWeakPtr < category)){
		category = job;
	}

	const Mutex::UniqueLock lock(g_mutex);
	AUTO_REF(list, g_fiberMap[category]);
	if(list.empty()){
		if(g_fiberPool.empty()){
			list.push_back(VAL_INIT);
		} else {
			list.splice(list.end(), g_fiberPool, g_fiberPool.begin());
		}
		list.back().state = FS_READY;
	}
	list.back().queue.push_back(JobElement(STD_MOVE(job), STD_MOVE(pred), STD_MOVE(withdrawn)));
}
void JobDispatcher::yield(boost::function<bool ()> pred){
	PROFILE_ME;

	AUTO_REF(fiber, g_currentFiber);
	if(!fiber){
		DEBUG_THROW(Exception, sslit("No current fiber"));
	}

	if(fiber->queue.empty()){
		LOG_POSEIDON_FATAL("Not in current fiber?!");
		std::abort();
	}
	fiber->queue.front().pred.swap(pred);

	LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_DEBUG, "Yielding from fiber ", static_cast<void *>(fiber));
	if(setjmp(fiber->inner) == 0){
		fiber->state = FS_YIELDED;
		std::longjmp(fiber->outer, 1);
	}
	LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_DEBUG, "Resumed to fiber ", static_cast<void *>(fiber));
}

void JobDispatcher::pumpAll(){
	LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_INFO, "Flushing all queued jobs...");

	reallyPumpJobs();
}

}
