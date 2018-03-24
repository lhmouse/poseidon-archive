// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2018, LH_Mouse. All wrongs reserved.

#include "../precompiled.hpp"
#include "job_dispatcher.hpp"
#include "main_config.hpp"
#include <ucontext.h>
#include <sys/mman.h>
#include "../job_base.hpp"
#include "../promise.hpp"
#include "../atomic.hpp"
#include "../exception.hpp"
#include "../log.hpp"
#include "../profiler.hpp"
#include "../mutex.hpp"
#include "../recursive_mutex.hpp"
#include "../condition_variable.hpp"
#include "../time.hpp"
#include "../checked_arithmetic.hpp"

namespace Poseidon {

namespace {
	enum FiberState {
		fiber_state_ready    = 0,
		fiber_state_running  = 1,
		fiber_state_yielded  = 2,
	};

	struct JobElement {
		boost::shared_ptr<JobBase> job;
		boost::shared_ptr<const bool> withdrawn;

		boost::shared_ptr<const Promise> promise;
		boost::uint64_t expiry_time;
		bool insignificant;
	};

	struct StackStorage FINAL {
		static void *operator new(std::size_t cb){
			assert(cb == sizeof(StackStorage));
			(void)cb;

			void *const ptr = mmap(NULLPTR, sizeof(StackStorage), PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_STACK, -1, 0);
			if(ptr == MAP_FAILED){
				const int err_code = errno;
				LOG_POSEIDON_ERROR("Failed to allocate stack: err_code = ", err_code);
				throw std::bad_alloc();
			}
			return ptr;
		}
		static void operator delete(void *ptr) NOEXCEPT {
			if(::munmap(ptr, sizeof(StackStorage)) != 0){
				const int err_code = errno;
				LOG_POSEIDON_ERROR("Failed to deallocate stack: err_code = ", err_code);
				std::abort();
			}
		}

		char bytes[0x40000];
	};

	class FiberStackAllocator : NONCOPYABLE {
	private:
		mutable Mutex m_mutex;
		boost::array<boost::scoped_ptr<StackStorage>, 64> m_pool;
		std::size_t m_size;

	public:
		FiberStackAllocator()
			: m_mutex(), m_pool(), m_size(0)
		{
			//
		}

	public:
		void allocate(boost::scoped_ptr<StackStorage> &ptr){
			const Mutex::UniqueLock lock(m_mutex);
			if(m_size == 0){
				ptr.reset(new StackStorage);
			} else {
				ptr.swap(m_pool.at(--m_size));
			}
		}
		void deallocate(boost::scoped_ptr<StackStorage> &ptr) NOEXCEPT {
			const Mutex::UniqueLock lock(m_mutex);
			if(m_size == m_pool.size()){
				ptr.reset();
			} else {
				ptr.swap(m_pool.at(m_size++));
			}
		}
	} g_stack_allocator;

	struct FiberControl : NONCOPYABLE {
		struct Initializer { };

		RecursiveMutex queue_mutex;
		boost::container::deque<JobElement> queue;

		FiberState state;
		boost::scoped_ptr<StackStorage> stack;
		::ucontext_t inner;
		::ucontext_t outer;

		explicit FiberControl(Initializer){
			state = fiber_state_ready;
			g_stack_allocator.allocate(stack);
#ifndef NDEBUG
			std::memset(&inner, 0xCC, sizeof(outer));
			std::memset(&outer, 0xCC, sizeof(outer));
#endif
		}
		~FiberControl(){
			assert(state == fiber_state_ready);
			g_stack_allocator.deallocate(stack);
#ifndef NDEBUG
			std::memset(&inner, 0xCC, sizeof(outer));
			std::memset(&outer, 0xCC, sizeof(outer));
#endif
		}
	};

	__thread FiberControl *volatile t_current_fiber = 0; // XXX: NULLPTR

	Mutex g_fiber_map_mutex;
	ConditionVariable g_new_job;
	boost::container::map<boost::weak_ptr<const void>, FiberControl> g_fiber_map;

	void fiber_proc(int low, int high) NOEXCEPT {
		PROFILE_ME;

		FiberControl *fiber;
		const int params[2] = { low, high };
		std::memcpy(&fiber, params, sizeof(fiber));

		LOG_POSEIDON_TRACE("Entering fiber ", static_cast<void *>(fiber));
		try {
			fiber->queue.front().job->perform();
		} catch(std::exception &e){
			LOG_POSEIDON_WARNING("std::exception thrown: what = ", e.what());
		} catch(...){
			LOG_POSEIDON_WARNING("Unknown exception thrown");
		}
		LOG_POSEIDON_TRACE("Exited from fiber ", static_cast<void *>(fiber));

		fiber->state = fiber_state_ready;
	}

	void schedule_fiber(FiberControl *fiber) NOEXCEPT {
		PROFILE_ME;

		if(fiber->state == fiber_state_ready){
			if(::getcontext(&(fiber->inner)) != 0){
				const int err_code = errno;
				LOG_POSEIDON_FATAL("::getcontext() failed: err_code = ", err_code);
				std::abort();
			}
			fiber->inner.uc_stack.ss_sp = fiber->stack.get();
			fiber->inner.uc_stack.ss_size = sizeof(*(fiber->stack));
			fiber->inner.uc_link = &(fiber->outer);

			int params[2] = { };
			BOOST_STATIC_ASSERT(sizeof(fiber) <= sizeof(params));
			std::memcpy(params, &fiber, sizeof(fiber));
			::makecontext(&(fiber->inner), reinterpret_cast<void (*)()>(&fiber_proc), 2, params[0], params[1]);
		}

		t_current_fiber = fiber;
		const AUTO(profiler_hook, Profiler::begin_stack_switch());
		{
			if((fiber->state != fiber_state_ready) && (fiber->state != fiber_state_yielded)){
				LOG_POSEIDON_FATAL("Fiber can't be scheduled: state = ", static_cast<int>(fiber->state));
				std::abort();
			}
			fiber->state = fiber_state_running;
			if(::swapcontext(&(fiber->outer), &(fiber->inner)) != 0){
				const int err_code = errno;
				LOG_POSEIDON_FATAL("::swapcontext() failed: err_code = ", err_code);
				std::abort();
			}
		}
		Profiler::end_stack_switch(profiler_hook);
		t_current_fiber = NULLPTR;
	}

	bool pump_one_fiber(FiberControl *fiber, bool force_expiry) NOEXCEPT {
		PROFILE_ME;

		const AUTO(now, get_fast_mono_clock());

		JobElement *elem;
		{
			const RecursiveMutex::UniqueLock queue_lock(fiber->queue_mutex);
			if(fiber->queue.empty()){
				return false;
			}
			elem = &(fiber->queue.front());
			if(elem->promise && !elem->promise->is_satisfied()){
				if((now < elem->expiry_time) && !(elem->insignificant && force_expiry)){
					return false;
				}
				LOG_POSEIDON_WARNING("Job timed out");
			}
			elem->promise.reset();
		}
		if((fiber->state == fiber_state_ready) && elem->withdrawn && *(elem->withdrawn)){
			LOG_POSEIDON_DEBUG("Job is withdrawn");
		} else {
			schedule_fiber(fiber);
		}
		if(fiber->state == fiber_state_ready){
			const RecursiveMutex::UniqueLock queue_lock(fiber->queue_mutex);
			fiber->queue.pop_front();
		}
		return true;
	}
	bool pump_one_round(bool force_expiry) NOEXCEPT {
		PROFILE_ME;

		bool busy = false;
		Mutex::UniqueLock lock(g_fiber_map_mutex);
		AUTO(it, g_fiber_map.begin());
		for(;;){
			if(it == g_fiber_map.end()){
				break;
			}
			AUTO(fiber, &(it->second));
			lock.unlock();
			{
				busy += pump_one_fiber(fiber, force_expiry);
			}
			lock.lock();
			if(fiber->queue.empty()){
				it = g_fiber_map.erase(it);
			} else {
				++it;
			}
		}
		return busy;
	}
}

void JobDispatcher::start(){
	LOG_POSEIDON(Logger::special_major | Logger::level_info, "Starting job dispatcher...");

	//
}
void JobDispatcher::stop(){
	LOG_POSEIDON(Logger::special_major | Logger::level_info, "Stopping job dispatcher...");

	Mutex::UniqueLock lock(g_fiber_map_mutex);
	boost::uint64_t last_info_time = 0;
	for(;;){
		const AUTO(pending_fibers, g_fiber_map.size());
		if(pending_fibers == 0){
			break;
		}
		lock.unlock();

		const AUTO(now, get_fast_mono_clock());
		if(last_info_time + 500 < now){
			LOG_POSEIDON(Logger::special_major | Logger::level_info, "There are ", pending_fibers, " fiber(s) remaining.");
			last_info_time = now;
		}
		pump_one_round(true);

		lock.lock();
	}
}

void JobDispatcher::do_modal(const volatile bool &running){
	unsigned timeout = 0;
	for(;;){
		bool busy;
		do {
			busy = pump_one_round(!atomic_load(running, memory_order_consume));
			timeout = std::min(timeout * 2u + 1u, !busy * 100u);
		} while(busy);

		Mutex::UniqueLock lock(g_fiber_map_mutex);
		if(!atomic_load(running, memory_order_consume)){
			break;
		}
		g_new_job.timed_wait(lock, timeout);
	}
}

void JobDispatcher::enqueue(boost::shared_ptr<JobBase> job, boost::shared_ptr<const bool> withdrawn){
	PROFILE_ME;

	const boost::weak_ptr<const void> null_weak_ptr;
	AUTO(category, job->get_category());
	if(!(category < null_weak_ptr) && !(null_weak_ptr < category)){
		category = job;
	}

	const Mutex::UniqueLock lock(g_fiber_map_mutex);
	AUTO(it, g_fiber_map.find(category));
	if(it == g_fiber_map.end()){
		it = g_fiber_map.emplace(category, FiberControl::Initializer()).first;
	}
	const AUTO(fiber, &(it->second));
	{
		const RecursiveMutex::UniqueLock queue_lock(fiber->queue_mutex);
		JobElement elem = { STD_MOVE(job), STD_MOVE(withdrawn) };
		fiber->queue.push_back(STD_MOVE(elem));
	}
	g_new_job.signal();
}
void JobDispatcher::yield(boost::shared_ptr<const Promise> promise, bool insignificant){
	PROFILE_ME;

	const AUTO(fiber, t_current_fiber);
	DEBUG_THROW_UNLESS(fiber, Exception, sslit("No current fiber"));
	if(fiber->queue.empty()){
		LOG_POSEIDON_FATAL("Not in current fiber?!");
		std::abort();
	}
	if(promise && promise->is_satisfied()){
		LOG_POSEIDON_TRACE("Skipped yielding from fiber ", static_cast<void *>(fiber));
	} else {
		LOG_POSEIDON_TRACE("Yielding from fiber ", static_cast<void *>(fiber));
		const AUTO(job_timeout, MainConfig::get<boost::uint64_t>("job_timeout", 60000));
		AUTO_REF(elem, fiber->queue.front());
		elem.promise = promise;
		elem.expiry_time = saturated_add(get_fast_mono_clock(), job_timeout);
		elem.insignificant = insignificant;
		const AUTO(profiler_hook, Profiler::begin_stack_switch());
		{
			fiber->state = fiber_state_yielded;
			if(::swapcontext(&(fiber->inner), &(fiber->outer)) != 0){
				const int err_code = errno;
				LOG_POSEIDON_FATAL("::swapcontext() failed: err_code = ", err_code);
				std::abort();
			}
		}
		Profiler::end_stack_switch(profiler_hook);
		LOG_POSEIDON_TRACE("Resumed to fiber ", static_cast<void *>(fiber));
	}

	if(promise){
		promise->check_and_rethrow();
	}
}

}
