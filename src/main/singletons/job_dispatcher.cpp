#include "../../precompiled.hpp"
#include "job_dispatcher.hpp"
#include <boost/thread/mutex.hpp>
#include <boost/thread/condition_variable.hpp>
#include <list>
#include "../job_base.hpp"
#include "../atomic.hpp"
#include "../exception.hpp"
#include "../log.hpp"
#include "../profiler.hpp"
using namespace Poseidon;

namespace {

volatile bool g_running = false;

boost::mutex g_mutex;
std::list<boost::shared_ptr<JobBase> > g_queue;
std::list<boost::shared_ptr<JobBase> > g_pool;
boost::condition_variable g_newJobAvail;

}

void JobDispatcher::doModal(){
	if(atomicExchange(g_running, true) != false){
		LOG_FATAL("Only one modal loop is allowed at the same time.");
		std::abort();
	}
	for(;;){
		PROFILE_ME;

		try {
			boost::shared_ptr<JobBase> job;
			{
				boost::mutex::scoped_lock lock(g_mutex);
				for(;;){
					if(!g_queue.empty()){
						job.swap(g_queue.front());
						g_pool.splice(g_pool.begin(), g_queue, g_queue.begin());
						break;
					}
					if(!atomicLoad(g_running)){
						break;
					}
					g_newJobAvail.wait(lock);
				}
			}
			if(job){
				job->perform();
			} else {
				break;
			}
		} catch(std::exception &e){
			LOG_ERROR("std::exception thrown in job dispatcher: what = ", e.what());
		} catch(...){
			LOG_ERROR("Unknown exception thrown in job dispatcher.");
		}
	}
}
void JobDispatcher::quitModal(){
	atomicStore(g_running, false);
	const boost::mutex::scoped_lock lock(g_mutex);
	g_newJobAvail.notify_all();
}

void JobDispatcher::pend(boost::shared_ptr<JobBase> job){
	const boost::mutex::scoped_lock lock(g_mutex);
	if(g_pool.empty()){
		g_pool.push_front(NULLPTR);
	}
	g_queue.splice(g_queue.end(), g_pool, g_pool.begin());
	g_queue.back().swap(job);
	g_newJobAvail.notify_all();
}
