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
		FS_READY,
		FS_RUNNING,
		FS_YIELDED,
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

	typedef boost::array<char, 0x100000> StackStorage;

	Mutex g_stackPoolMutex;
	std::list<StackStorage> g_stackPool;

	struct FiberControl {
		std::deque<JobElement> queue;

		FiberState state;
		std::list<StackStorage> stack;
		std::jmp_buf outer;
		std::jmp_buf inner;

		FiberControl()
			: state(FS_READY)
		{
			const Mutex::UniqueLock lock(g_stackPoolMutex);
			if(g_stackPool.empty()){
				g_stackPool.resize(10);
			}
			stack.splice(stack.end(), g_stackPool, g_stackPool.begin());
		}
		FiberControl(const FiberControl &rhs) NOEXCEPT
			: state(FS_READY)
		{
			if((rhs.state != FS_READY) || !rhs.queue.empty()){
				std::abort();
			}

			const Mutex::UniqueLock lock(g_stackPoolMutex);
			if(g_stackPool.empty()){
				g_stackPool.resize(10);
			}
			stack.splice(stack.end(), g_stackPool, g_stackPool.begin());
		}
		~FiberControl(){
			const Mutex::UniqueLock lock(g_stackPoolMutex);
			g_stackPool.splice(g_stackPool.end(), stack);
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
			if(fiber->state == FS_YIELDED){
				fiber->state = FS_RUNNING;
				std::longjmp(fiber->inner, 1);
			} else {
				fiber->state = FS_RUNNING;
				LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_TRACE, "Entering fiber ", static_cast<void *>(fiber));
				{
					register void *const sp = fiber->stack.front().end();
					__asm__(
#ifdef __x86_64__
						"movq %%rax, %%rsp \n"
#else
						"movl %%eax, %%esp \n"
#endif
						: : "a"(sp) :
					);
				}
				try {
					fiber->queue.front().job->perform();
				} catch(std::exception &e){
					LOG_POSEIDON_WARNING("std::exception thrown: what = ", e.what());
				} catch(...){
					LOG_POSEIDON_WARNING("Unknown exception thrown");
				}
				LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_TRACE, "Exited from fiber ", static_cast<void *>(fiber));

				fiber->state = FS_READY;
				std::longjmp(fiber->outer, 1);
			}
		}

		g_currentFiber = NULLPTR;
	}

	volatile bool g_running = false;

	Mutex g_fiberMutex;
	ConditionVariable g_newJob;
	std::map<boost::weak_ptr<const void>, FiberControl> g_fiberMap;

	void reallyPumpJobs() NOEXCEPT {
		PROFILE_ME;

		bool busy;
		do {
			busy = false;

			Mutex::UniqueLock lock(g_fiberMutex);
			AUTO(it, g_fiberMap.begin());
			while(it != g_fiberMap.end()){
				if((it->second.state == FS_READY) && (it->second.queue.empty())){
					g_fiberMap.erase(it++);
					continue;
				}
				lock.unlock();

				AUTO_REF(elem, it->second.queue.front());
				bool done;

				if(elem.withdrawn && *elem.withdrawn){
					LOG_POSEIDON_DEBUG("Job is withdrawn");
					done = true;
				} else if(elem.pred && !elem.pred()){
					LOG_POSEIDON_TRACE("Job is not ready to be scheduled");
					done = false;
				} else {
					scheduleFiber(&it->second);
					done = (it->second.state == FS_READY);
				}

				lock.lock();
				if(done){
					it->second.queue.pop_front();
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
			const Mutex::UniqueLock lock(g_fiberMutex);
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

		Mutex::UniqueLock lock(g_fiberMutex);
		if(!atomicLoad(g_running, ATOMIC_CONSUME) && g_fiberMap.empty()){
			break;
		}
		g_newJob.timedWait(lock, 100);
	}
}
bool JobDispatcher::isRunning(){
	return atomicLoad(g_running, ATOMIC_CONSUME);
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

	const Mutex::UniqueLock lock(g_fiberMutex);
	AUTO_REF(queue, g_fiberMap[category].queue);
	queue.push_back(JobElement(STD_MOVE(job), STD_MOVE(pred), STD_MOVE(withdrawn)));
	g_newJob.signal();
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

	LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_TRACE, "Yielding from fiber ", static_cast<void *>(fiber));
	if(setjmp(fiber->inner) == 0){
		fiber->state = FS_YIELDED;
		std::longjmp(fiber->outer, 1);
	}
	LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_TRACE, "Resumed to fiber ", static_cast<void *>(fiber));
}

void JobDispatcher::pumpAll(){
	LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_INFO, "Flushing all queued jobs...");

	reallyPumpJobs();
}

}
