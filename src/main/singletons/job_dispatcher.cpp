// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2015, LH_Mouse. All wrongs reserved.

#include "../precompiled.hpp"
#include "job_dispatcher.hpp"
#include <boost/thread/mutex.hpp>
#include <boost/thread/condition_variable.hpp>
#include "../job_base.hpp"
#include "../atomic.hpp"
#include "../exception.hpp"
#include "../log.hpp"
#include "../profiler.hpp"

namespace Poseidon {

namespace {
	volatile bool g_running = false;

	boost::mutex g_mutex;
	std::deque<boost::shared_ptr<JobBase> > g_queue;
	boost::condition_variable g_newJobAvail;
}

void JobDispatcher::start(){
}
void JobDispatcher::stop(){
	LOG_POSEIDON_INFO("Flushing all pending jobs...");

	for(;;){
		std::deque<boost::shared_ptr<JobBase> > queue;
		{
			const boost::mutex::scoped_lock lock(g_mutex);
			queue.swap(g_queue);
		}
		if(queue.empty()){
			break;
		}

		AUTO(it, queue.begin());
		do {
			try {
				(*it)->perform();
			} catch(std::exception &e){
				LOG_POSEIDON_ERROR("std::exception thrown in job dispatcher: what = ", e.what());
			} catch(...){
				LOG_POSEIDON_ERROR("Unknown exception thrown in job dispatcher.");
			}
			it->reset();
		} while(++it != queue.end());
	}
}

void JobDispatcher::doModal(){
	LOG_POSEIDON_INFO("Entering modal loop...");

	if(atomicExchange(g_running, true, ATOMIC_ACQ_REL) != false){
		LOG_POSEIDON_FATAL("Only one modal loop is allowed at the same time.");
		std::abort();
	}

	for(;;){
		std::deque<boost::shared_ptr<JobBase> > queue;
		{
			boost::mutex::scoped_lock lock(g_mutex);
			while(g_queue.empty()){
				if(!atomicLoad(g_running, ATOMIC_ACQUIRE)){
					break;
				}
				g_newJobAvail.wait(lock);
			}
			queue.swap(g_queue);
		}
		if(queue.empty()){
			break;
		}

		AUTO(it, queue.begin());
		do {
			try {
				(*it)->perform();
			} catch(std::exception &e){
				LOG_POSEIDON_ERROR("std::exception thrown in job dispatcher: what = ", e.what());
			} catch(...){
				LOG_POSEIDON_ERROR("Unknown exception thrown in job dispatcher.");
			}
			it->reset();
		} while(++it != queue.end());
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

void JobDispatcher::pend(boost::shared_ptr<JobBase> job){
	const boost::mutex::scoped_lock lock(g_mutex);
	g_queue.push_back(STD_MOVE(job));
	g_newJobAvail.notify_all();
}

}
