// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2015, LH_Mouse. All wrongs reserved.

#include "../precompiled.hpp"
#include "job_dispatcher.hpp"
#include "main_config.hpp"
#include <stdlib.h>
#include <setjmp.h>
#include "../job_base.hpp"
#include "../atomic.hpp"
#include "../exception.hpp"
#include "../log.hpp"
#include "../profiler.hpp"
#include "../mutex.hpp"
#include "../condition_variable.hpp"
#include "../time.hpp"

extern "C"
__attribute__((__noreturn__, __nothrow__))
void unchecked_siglongjmp(::sigjmp_buf, int)
	__asm__("siglongjmp");

namespace {

__attribute__((__noinline__))
void dont_optimize_try_catch_away();

}

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

	struct StackStorage FINAL {
		static void *operator new(std::size_t cb){
			assert(cb == sizeof(StackStorage));

			void *ptr;
			const int errCode = ::posix_memalign(&ptr, 0x1000, cb);
			if(errCode != 0){
				LOG_POSEIDON_WARNING("Failed to allocate stack: errCode = ", errCode);
				throw std::bad_alloc();
			}
			return ptr;
		}
		static void operator delete(void *ptr) NOEXCEPT {
			::free(ptr);
		}

		char bytes[0x100000];
	};

	Mutex g_stackPoolMutex;
	boost::array<boost::scoped_ptr<StackStorage>, 16> g_stackPool;

	struct FiberControl {
		std::deque<JobElement> queue;

		FiberState state;
		boost::scoped_ptr<StackStorage> stack;
		::sigjmp_buf outer;
		::sigjmp_buf inner;

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
		~FiberControl(){
			if(stack){
				const Mutex::UniqueLock lock(g_stackPoolMutex);
				for(AUTO(it, g_stackPool.begin()); it != g_stackPool.end(); ++it){
					if(!*it){
						stack.swap(*it);
						break;
					}
				}
			}
		}
	};

	__attribute__((__noinline__, __nothrow__))
#ifndef __x86_64__
	__attribute__((__force_align_arg_pointer__))
#endif
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
		try {
			::dont_optimize_try_catch_away();

			if(sigsetjmp(fiber->outer, true) == 0){
				if(fiber->state == FS_YIELDED){
					fiber->state = FS_RUNNING;
					::unchecked_siglongjmp(fiber->inner, 1);
				} else {
					fiber->state = FS_RUNNING;
					if(!fiber->stack){
						const Mutex::UniqueLock lock(g_stackPoolMutex);
						for(AUTO(it, g_stackPool.begin()); it != g_stackPool.end(); ++it){
							if(*it){
								fiber->stack.swap(*it);
								goto _allocated;
							}
						}
						fiber->stack.reset(new StackStorage);
					_allocated:
						;
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
						: "c"(fiber), "a"(fiber->stack.get() + 1) // sp 指向可用栈区的末尾。
						: "sp", "memory"
					);
					fiberStackBarrier(reg);

					reg->state = FS_READY;
					::unchecked_siglongjmp(reg->outer, 1);
				}
			}

			::dont_optimize_try_catch_away();
		} catch(...){
			std::abort();
		}
		t_currentFiber = NULLPTR;
	}

	volatile bool g_running = false;

	Mutex g_fiberMutex;
	ConditionVariable g_newJob;
	std::map<boost::weak_ptr<const void>, FiberControl> g_fiberMap;

	void reallyPumpJobs() NOEXCEPT {
		PROFILE_ME;

		Mutex::UniqueLock lock(g_fiberMutex);
		for(AUTO(next, g_fiberMap.begin()), it = next; (next != g_fiberMap.end()) && (++next, true); it = next){
			bool done = true;
			unsigned count = 0;
			for(;;){
				if(it->second.queue.empty()){
					g_fiberMap.erase(it);
					break;
				}
				if(!done){
					break;
				}
				if(count >= 10){
					break;
				}
				lock.unlock();

				AUTO_REF(elem, it->second.queue.front());
				if((it->second.state == FS_READY) && elem.withdrawn && *elem.withdrawn){
					LOG_POSEIDON_DEBUG("Job is withdrawn");
					done = true;
				} else if(elem.pred && !elem.pred()){
					done = false;
				} else {
					boost::function<bool ()>().swap(elem.pred);
					scheduleFiber(&it->second);
					done = (it->second.state == FS_READY);
				}

				lock.lock();
				if(done){
					it->second.queue.pop_front();
				}
				++count;
			}
		}
	}
}

void JobDispatcher::start(){
	LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_INFO, "Starting job dispatcher...");

//	MainConfig::get(g_maxRetryCount, "job_max_retry_count");
//	LOG_POSEIDON_DEBUG("Max retry count = ", g_maxRetryCount);
}
void JobDispatcher::stop(){
	LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_INFO, "Stopping job dispatcher...");

	boost::uint64_t lastInfoTime = 0;
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
	g_newJob.signal();
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
	try {
		::dont_optimize_try_catch_away();

		if(sigsetjmp(fiber->inner, true) == 0){
			fiber->state = FS_YIELDED;
			::unchecked_siglongjmp(fiber->outer, 1);
		}

		::dont_optimize_try_catch_away();
	} catch(...){
		std::abort();
	}
	LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_TRACE, "Resumed to fiber ", static_cast<void *>(fiber));
}
void JobDispatcher::detachYieldable() NOEXCEPT {
	PROFILE_ME;

	t_currentFiber = NULLPTR;
}

void JobDispatcher::pumpAll(){
	LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_INFO, "Flushing all queued jobs...");

	reallyPumpJobs();
}

}

namespace {

__attribute__((__noinline__))
void dont_optimize_try_catch_away(){
}

}
