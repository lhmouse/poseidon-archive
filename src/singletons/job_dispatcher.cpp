// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2016, LH_Mouse. All wrongs reserved.

#include "../precompiled.hpp"
#include "job_dispatcher.hpp"
#include "main_config.hpp"
#include <stdlib.h>
#include <setjmp.h>
#include "../job_base.hpp"
#include "../job_promise.hpp"
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
	boost::uint64_t g_job_timeout = 60000;

	enum FiberState {
		FS_READY,
		FS_RUNNING,
		FS_YIELDED,
	};

	struct JobElement {
		boost::shared_ptr<JobBase> job;
		boost::shared_ptr<const JobPromise> promise;
		boost::uint64_t expiry_time;
		boost::shared_ptr<const bool> withdrawn;

		JobElement(boost::shared_ptr<JobBase> job_, boost::shared_ptr<const JobPromise> promise_,
			boost::shared_ptr<const bool> withdrawn_)
			: job(STD_MOVE(job_)), promise(STD_MOVE(promise_)), expiry_time(get_fast_mono_clock() + g_job_timeout)
			, withdrawn(STD_MOVE(withdrawn_))
		{
		}
	};

	struct StackStorage FINAL {
		static void *operator new(std::size_t cb){
			assert(cb == sizeof(StackStorage));

			void *ptr;
			const int err_code = ::posix_memalign(&ptr, 0x1000, cb);
			if(err_code != 0){
				LOG_POSEIDON_WARNING("Failed to allocate stack: err_code = ", err_code);
				throw std::bad_alloc();
			}
			return ptr;
		}
		static void operator delete(void *ptr) NOEXCEPT {
			::free(ptr);
		}

		char bytes[0x100000];
	};

	boost::array<boost::scoped_ptr<StackStorage>, 16> g_stack_pool;

	struct FiberControl {
		std::deque<JobElement> queue;

		FiberState state;
		boost::scoped_ptr<StackStorage> stack;
		::sigjmp_buf outer;
		::sigjmp_buf inner;
		void *profiler_hook;

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
#ifndef __x86_64__
		__attribute__((__force_align_arg_pointer__))
#endif
	void fiber_stack_barrier(FiberControl *fiber) NOEXCEPT {
		PROFILE_ME;

		LOG_POSEIDON_TRACE("Entering fiber ", static_cast<void *>(fiber));
		try {
			const AUTO_REF(elem, fiber->queue.front());
			elem.job->perform();
		} catch(std::exception &e){
			LOG_POSEIDON_WARNING("std::exception thrown: what = ", e.what());
		} catch(...){
			LOG_POSEIDON_WARNING("Unknown exception thrown");
		}
		LOG_POSEIDON_TRACE("Exited from fiber ", static_cast<void *>(fiber));
	}

	AUTO(g_current_fiber, (FiberControl *)0);

	__attribute__((__noinline__, __nothrow__))
	void schedule_fiber(FiberControl *fiber) NOEXCEPT {
		PROFILE_ME;

		assert(!fiber->queue.empty());

		if((fiber->state != FS_READY) && (fiber->state != FS_YIELDED)){
			LOG_POSEIDON_FATAL("Fiber can't be scheduled: state = ", static_cast<int>(fiber->state));
			std::abort();
		}

		g_current_fiber = fiber;
		try {
			::dont_optimize_try_catch_away();

			const AUTO(profiler_hook, Profiler::begin_stack_switch());

			if(sigsetjmp(fiber->outer, true) == 0){
				if(fiber->state == FS_YIELDED){
					fiber->state = FS_RUNNING;
					::unchecked_siglongjmp(fiber->inner, 1);
				} else {
					fiber->state = FS_RUNNING;
					if(!fiber->stack){
						for(AUTO(it, g_stack_pool.begin()); it != g_stack_pool.end(); ++it){
							if(*it){
								fiber->stack.swap(*it);
								goto _allocated;
							}
						}
						fiber->stack.reset(new StackStorage);
					_allocated:
						;
					}

					register AUTO(reg __asm__("bx"), fiber); // 必须是 callee-saved 寄存器！
					__asm__ __volatile__(
#ifdef __x86_64__
						"movq %%rax, %%rsp \n"
#else
						"movl %%eax, %%esp \n"
#endif
						: "+b"(reg)
						: "a"(fiber->stack.get() + 1) // sp 指向可用栈区的末尾。
						: "sp", "memory"
					);
					fiber_stack_barrier(reg);

					reg->state = FS_READY;
					::unchecked_siglongjmp(reg->outer, 1);
				}
			}

			Profiler::end_stack_switch(profiler_hook);

			::dont_optimize_try_catch_away();
		} catch(...){
			std::abort();
		}
		g_current_fiber = NULLPTR;
	}

	volatile bool g_running = false;

	Mutex g_fiber_mutex;
	ConditionVariable g_new_job;
	std::map<boost::weak_ptr<const void>, FiberControl> g_fiber_map;

	void really_pump_jobs() NOEXCEPT {
		PROFILE_ME;

		AUTO(now, get_fast_mono_clock());

		Mutex::UniqueLock lock(g_fiber_mutex);
		bool busy;
		do {
			busy = false;

			for(AUTO(next, g_fiber_map.begin()), it = next; (next != g_fiber_map.end()) && (++next, true); it = next){
				for(;;){
					if(it->second.queue.empty()){
						for(AUTO(pit, g_stack_pool.begin()); pit != g_stack_pool.end(); ++pit){
							if(!*pit){
								pit->swap(it->second.stack);
								break;
							}
						}
						g_fiber_map.erase(it);
						break;
					}
					AUTO_REF(elem, it->second.queue.front());
					if(elem.promise && !elem.promise->is_satisfied()){
						if(now < elem.expiry_time){
							break;
						}
						LOG_POSEIDON_ERROR("Job timed out");
					}
					lock.unlock();

					elem.promise.reset();
					busy = true;

					bool done;
					if((it->second.state == FS_READY) && elem.withdrawn && *elem.withdrawn){
						LOG_POSEIDON_DEBUG("Job is withdrawn");
						done = true;
					} else {
						schedule_fiber(&it->second);
						done = (it->second.state == FS_READY);
					}

					now = get_fast_mono_clock();

					lock.lock();
					if(done){
						it->second.queue.pop_front();
					}
				}
			}
		} while(busy);
	}
}

void JobDispatcher::start(){
	LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_INFO, "Starting job dispatcher...");

	MainConfig::get(g_job_timeout, "job_timeout");
	LOG_POSEIDON_DEBUG("Job timeout = ", g_job_timeout);
}
void JobDispatcher::stop(){
	LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_INFO, "Stopping job dispatcher...");

	boost::uint64_t last_info_time = 0;
	for(;;){
		std::size_t pending_fibers;
		{
			const Mutex::UniqueLock lock(g_fiber_mutex);
			pending_fibers = g_fiber_map.size();
		}
		if(pending_fibers == 0){
			break;
		}

		const AUTO(now, get_fast_mono_clock());
		if(last_info_time + 500 < now){
			LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_INFO, "There are ", pending_fibers, " fiber(s) remaining.");
			last_info_time = now;
		}

		really_pump_jobs();
	}
}

void JobDispatcher::do_modal(){
	LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_INFO, "Entering modal loop...");

	if(atomic_exchange(g_running, true, ATOMIC_ACQ_REL) != false){
		LOG_POSEIDON_FATAL("Only one modal loop is allowed at the same time.");
		std::abort();
	}

	for(;;){
		really_pump_jobs();

		Mutex::UniqueLock lock(g_fiber_mutex);
		if(!atomic_load(g_running, ATOMIC_CONSUME)){
			break;
		}
		g_new_job.timed_wait(lock, 100);
	}
}
bool JobDispatcher::is_running(){
	return atomic_load(g_running, ATOMIC_CONSUME);
}
void JobDispatcher::quit_modal(){
	atomic_store(g_running, false, ATOMIC_RELEASE);
	g_new_job.signal();
}

void JobDispatcher::enqueue(boost::shared_ptr<JobBase> job,
	boost::shared_ptr<const JobPromise> promise, boost::shared_ptr<const bool> withdrawn)
{
	PROFILE_ME;

	const boost::weak_ptr<const void> null_weak_ptr;
	AUTO(category, job->get_category());
	if(!(category < null_weak_ptr) && !(null_weak_ptr < category)){
		category = job;
	}

	const Mutex::UniqueLock lock(g_fiber_mutex);
	AUTO_REF(queue, g_fiber_map[category].queue);
	queue.push_back(JobElement(STD_MOVE(job), STD_MOVE(promise), STD_MOVE(withdrawn)));
	g_new_job.signal();
}
void JobDispatcher::yield(boost::shared_ptr<const JobPromise> promise){
	PROFILE_ME;

	const AUTO(fiber, g_current_fiber);
	if(!fiber){
		DEBUG_THROW(Exception, sslit("No current fiber"));
	}

	if(fiber->queue.empty()){
		LOG_POSEIDON_FATAL("Not in current fiber?!");
		std::abort();
	}
	AUTO_REF(elem, fiber->queue.front());
	elem.promise = promise;
	elem.expiry_time = get_fast_mono_clock() + g_job_timeout;

	LOG_POSEIDON_TRACE("Yielding from fiber ", static_cast<void *>(fiber));
	const AUTO(opaque, Profiler::begin_stack_switch());
	try {
		::dont_optimize_try_catch_away();

		if(sigsetjmp(fiber->inner, true) == 0){
			fiber->state = FS_YIELDED;
			fiber->profiler_hook = Profiler::begin_stack_switch();
			::unchecked_siglongjmp(fiber->outer, 1);
		}
		Profiler::end_stack_switch(fiber->profiler_hook);

		::dont_optimize_try_catch_away();
	} catch(...){
		std::abort();
	}
	Profiler::end_stack_switch(opaque);
	LOG_POSEIDON_TRACE("Resumed to fiber ", static_cast<void *>(fiber));

	if(promise){
		promise->check_and_rethrow();
	}
}

void JobDispatcher::pump_all(){
	LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_INFO, "Flushing all queued jobs...");

	really_pump_jobs();
}

}

namespace {

__attribute__((__noinline__))
void dont_optimize_try_catch_away(){
}

}
