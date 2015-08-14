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

	struct FiberControl {
		std::deque<JobElement> queue;

		FiberState state;
		boost::scoped_ptr<StackStorage> stack;
		std::jmp_buf outer;
		std::jmp_buf inner;

		FiberControl()
			: state(FS_READY)
		{
		}
		FiberControl(const FiberControl &rhs) NOEXCEPT
			: state(FS_READY)
		{
			if((rhs.state != FS_READY) || !rhs.queue.empty()){
				std::abort();
			}
		}
	};

	__attribute__((__noinline__, __nothrow__))
	void fiberStackBarrier(FiberControl *fiber) NOEXCEPT {
		LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_TRACE, "Entering fiber ", static_cast<void *>(fiber));
		try {
			fiber->queue.front().job->perform();
		} catch(std::exception &e){
			LOG_POSEIDON_WARNING("std::exception thrown: what = ", e.what());
		} catch(...){
			LOG_POSEIDON_WARNING("Unknown exception thrown");
		}
		LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_TRACE, "Exited from fiber ", static_cast<void *>(fiber));
	}

	__thread FiberControl *t_currentFiber = NULLPTR;

	void scheduleFiber(FiberControl *fiber) NOEXCEPT {
		PROFILE_ME;

		assert(!fiber->queue.empty());

		if((fiber->state != FS_READY) && (fiber->state != FS_YIELDED)){
			LOG_POSEIDON_FATAL("Fiber can't be scheduled: state = ", static_cast<int>(fiber->state));
			std::abort();
		}

		t_currentFiber = fiber;
		if(setjmp(fiber->outer) == 0){
			if(fiber->state == FS_YIELDED){
				fiber->state = FS_RUNNING;
				std::longjmp(fiber->inner, 1);
			} else {
				fiber->state = FS_RUNNING;
				if(!fiber->stack){
					fiber->stack.reset(new StackStorage);
				}

				register FiberControl *reg __asm__("bx");
				__asm__ __volatile__(
#ifdef __x86_64__
					"movq %%rcx, %%rbx \n"
					"movq %%rax, %%rsp \n"
#else
					"movl %%ecx, %%ebx \n"
					"movl %%eax, %%esp \n"
#endif
					: "=b"(reg)
					: "c"(fiber), "a"(fiber->stack->end())
					: "sp", "memory"
				);
				fiberStackBarrier(reg);

				reg->state = FS_READY;
				std::longjmp(reg->outer, 1);
			}
		}
		t_currentFiber = NULLPTR;
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
				} else {
					busy = true;

					if(!elem.pred || elem.pred()){
						boost::function<bool ()>().swap(elem.pred);

						scheduleFiber(&it->second);
						done = (it->second.state == FS_READY);
					} else {
						done = false;
					}
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
		std::size_t pendingFibers;
		{
			const Mutex::UniqueLock lock(g_fiberMutex);
			pendingFibers = g_fiberMap.size();
		}
		if(pendingFibers == 0){
			break;
		}

		const AUTO(now, getFastMonoClock());
		if(lastInfoTime + 500 < now){
			LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_INFO, "There are ", pendingFibers, " fiber(s) remaining.");
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

	const AUTO(fiber, t_currentFiber);
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
