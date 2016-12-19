// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2016, LH_Mouse. All wrongs reserved.

#include "../precompiled.hpp"
#include "job_dispatcher.hpp"
#include "main_config.hpp"
#include <ucontext.h>
#include <sys/mman.h>
#include <errno.h>
#include <boost/container/map.hpp>
#include <vector>
#include "../job_base.hpp"
#include "../job_promise.hpp"
#include "../atomic.hpp"
#include "../exception.hpp"
#include "../log.hpp"
#include "../profiler.hpp"
#include "../mutex.hpp"
#include "../recursive_mutex.hpp"
#include "../condition_variable.hpp"
#include "../time.hpp"

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
		boost::shared_ptr<const bool> withdrawn;

		boost::shared_ptr<const JobPromise> promise;
		boost::uint64_t expiry_time;
		bool insignificant;

		JobElement(boost::shared_ptr<JobBase> job_, boost::shared_ptr<const bool> withdrawn_)
			: job(STD_MOVE(job_)), withdrawn(STD_MOVE(withdrawn_))
			, promise(), expiry_time((boost::uint64_t)-1), insignificant(false)
		{
		}
	};

	class FiberStackAllocator : NONCOPYABLE {
	public:
		struct Storage FINAL {
			static void *operator new(std::size_t cb){
				assert(cb == sizeof(Storage));
				(void)cb;

				const AUTO(ptr, mmap(NULLPTR, sizeof(Storage),
					PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_GROWSDOWN | MAP_STACK, -1, 0));
				if(ptr == MAP_FAILED){
					const int err_code = errno;
					LOG_POSEIDON_ERROR("Failed to allocate stack: err_code = ", err_code);
					throw std::bad_alloc();
				}
				return ptr;
			}
			static void operator delete(void *ptr) NOEXCEPT {
				if(::munmap(ptr, sizeof(Storage)) != 0){
					const int err_code = errno;
					LOG_POSEIDON_ERROR("Failed to deallocate stack: err_code = ", err_code);
					std::abort();
				}
			}

			char bytes[256 * 1024];
		};

#ifdef POSEIDON_CXX11
		typedef std::unique_ptr<Storage> StoragePtr;
#else
		typedef boost::shared_ptr<Storage> StoragePtr;
#endif

	private:
		mutable Mutex m_mutex;
		boost::array<StoragePtr, 64> m_pool;
		std::size_t m_size;

	public:
		FiberStackAllocator()
			: m_mutex(), m_pool(), m_size(0)
		{
		}

	public:
		StoragePtr allocate(){
			StoragePtr ptr;
			const Mutex::UniqueLock lock(m_mutex);
			if(m_size == 0){
				ptr.reset(new Storage);
			} else {
				ptr = STD_MOVE(m_pool.at(--m_size));
			}
			return ptr;
		}
		void deallocate(StoragePtr ptr) NOEXCEPT {
			const Mutex::UniqueLock lock(m_mutex);
			if(m_size == m_pool.size()){
				ptr.reset();
			} else {
				m_pool.at(m_size++) = STD_MOVE(ptr);
			}
		}
	} g_stack_allocator;

	struct FiberControl : NONCOPYABLE {
		struct Initializer { };

		RecursiveMutex queue_mutex;
		std::deque<JobElement> queue;

		FiberState state;
		FiberStackAllocator::StoragePtr stack;
		::ucontext_t inner;
		::ucontext_t outer;

		explicit FiberControl(Initializer){
			state = FS_READY;
			stack = g_stack_allocator.allocate();
#ifndef NDEBUG
			std::memset(&inner, 0xCC, sizeof(outer));
			std::memset(&outer, 0xCC, sizeof(outer));
#endif
		}
		~FiberControl(){
			assert(state == FS_READY);
			g_stack_allocator.deallocate(STD_MOVE_IDN(stack));
#ifndef NDEBUG
			std::memset(&inner, 0xCC, sizeof(outer));
			std::memset(&outer, 0xCC, sizeof(outer));
#endif
		}
	};

	AUTO(g_current_fiber, (FiberControl *)0);

	volatile bool g_running = false;

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

		fiber->state = FS_READY;
	}

	void schedule_fiber(FiberControl *fiber) NOEXCEPT {
		PROFILE_ME;

		if(fiber->state == FS_READY){
			if(::getcontext(&(fiber->inner)) != 0){
				const int err_code = errno;
				LOG_POSEIDON_FATAL("::getcontext() failed: err_code = ", err_code);
				std::abort();
			}
			fiber->inner.uc_stack.ss_sp = fiber->stack.get();
			fiber->inner.uc_stack.ss_size = sizeof(*(fiber->stack));
			fiber->inner.uc_link = &(fiber->outer);

			int params[2] = { };
			std::memcpy(params, &fiber, sizeof(fiber));
			::makecontext(&(fiber->inner), reinterpret_cast<void (*)()>(&fiber_proc), 2, params[0], params[1]);
		}

		g_current_fiber = fiber;
		const AUTO(profiler_hook, Profiler::begin_stack_switch());
		{
			if((fiber->state != FS_READY) && (fiber->state != FS_YIELDED)){
				LOG_POSEIDON_FATAL("Fiber can't be scheduled: state = ", static_cast<int>(fiber->state));
				std::abort();
			}
			fiber->state = FS_RUNNING;
			if(::swapcontext(&(fiber->outer), &(fiber->inner)) != 0){
				const int err_code = errno;
				LOG_POSEIDON_FATAL("::swapcontext() failed: err_code = ", err_code);
				std::abort();
			}
		}
		Profiler::end_stack_switch(profiler_hook);
		g_current_fiber = NULLPTR;
	}

	bool really_pump_fiber(FiberControl *fiber) NOEXCEPT {
		PROFILE_ME;

		bool busy = false;
		for(;;){
			const AUTO(now, get_fast_mono_clock());

			JobElement *elem;
			{
				const RecursiveMutex::UniqueLock queue_lock(fiber->queue_mutex);
				if(fiber->queue.empty()){
					break;
				}
				elem = &(fiber->queue.front());
				if(elem->promise && !elem->promise->is_satisfied()){
					if((now < elem->expiry_time) && !(elem->insignificant && !atomic_load(g_running, ATOMIC_CONSUME))){
						break;
					}
					LOG_POSEIDON_WARNING("Job timed out");
				}
				elem->promise.reset();
			}
			++busy;

			try {
				if((fiber->state == FS_READY) && elem->withdrawn && *(elem->withdrawn)){
					LOG_POSEIDON_DEBUG("Job is withdrawn");
				} else {
					schedule_fiber(fiber);
				}
			} catch(...){
				std::abort();
			}
			if(fiber->state != FS_READY){
				break;
			}

			const RecursiveMutex::UniqueLock queue_lock(fiber->queue_mutex);
			fiber->queue.pop_front();
		}
		return busy;
	}

	bool really_pump_jobs() NOEXCEPT {
		PROFILE_ME;

		bool busy = false;
		Mutex::UniqueLock lock(g_fiber_map_mutex);
		AUTO(it, g_fiber_map.begin());
		while(it != g_fiber_map.end()){
			const AUTO(fiber, &(it->second));
			if(fiber->queue.empty()){
				it = g_fiber_map.erase(it);
				continue;
			}
			lock.unlock();

			busy += really_pump_fiber(fiber);

			lock.lock();
			++it;
		}
		return busy;
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
			const Mutex::UniqueLock lock(g_fiber_map_mutex);
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
		while(really_pump_jobs()){
			// noop
		}

		Mutex::UniqueLock lock(g_fiber_map_mutex);
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
		fiber->queue.push_back(JobElement(STD_MOVE(job), STD_MOVE(withdrawn)));
	}
	g_new_job.signal();
}
void JobDispatcher::yield(const boost::shared_ptr<const JobPromise> &promise, bool insignificant){
	PROFILE_ME;

	const AUTO(fiber, g_current_fiber);
	if(!fiber){
		DEBUG_THROW(Exception, sslit("No current fiber"));
	}

	const AUTO(now, get_fast_mono_clock());

	if(fiber->queue.empty()){
		LOG_POSEIDON_FATAL("Not in current fiber?!");
		std::abort();
	}
	AUTO_REF(elem, fiber->queue.front());
	elem.promise = promise;
	elem.expiry_time = now + g_job_timeout;
	elem.insignificant = insignificant;

	LOG_POSEIDON_TRACE("Yielding from fiber ", static_cast<void *>(fiber));
	const AUTO(profiler_hook, Profiler::begin_stack_switch());
	{
		fiber->state = FS_YIELDED;
		if(::swapcontext(&(fiber->inner), &(fiber->outer)) != 0){
			const int err_code = errno;
			LOG_POSEIDON_FATAL("::swapcontext() failed: err_code = ", err_code);
			std::abort();
		}
	}
	Profiler::end_stack_switch(profiler_hook);
	LOG_POSEIDON_TRACE("Resumed to fiber ", static_cast<void *>(fiber));

	if(promise){
		promise->check_and_rethrow();
	}
}

}
