#include "../../precompiled.hpp"
#include "job_dispatcher.hpp"
#include <boost/thread/mutex.hpp>
#include <boost/thread/condition_variable.hpp>
#include <list>
#include "../job_base.hpp"
#include "../atomic.hpp"
#include "../exception.hpp"
#include "../log.hpp"
using namespace Poseidon;

namespace {

volatile bool g_running = false;

boost::mutex g_mutex;
std::list<boost::shared_ptr<const JobBase> > g_queue;
std::list<boost::shared_ptr<const JobBase> > g_pool;

boost::condition_variable g_newJobAvail;

}

void JobDispatcher::doModal(){
	if(atomicExchange(g_running, true) != false){
		LOG_FATAL("Only one modal loop is allowed at the same time.");
		std::abort();
	}
	for(;;){
		try {
			boost::shared_ptr<const JobBase> job;
			{
				boost::mutex::scoped_lock lock(g_mutex);
				for(;;){
					if(!g_queue.empty()){
						job.swap(g_queue.front());
						g_pool.splice(g_pool.end(), g_queue, g_queue.begin());
						break;
					}
					if(!atomicLoad(g_running)){
						break;
					}
					g_newJobAvail.wait(lock);
				}
			}
			if(!job){
				break;
			}
			job->perform();
		} catch(Exception &e){
			LOG_ERROR("Exception thrown in job dispatcher: file = ", e.file(),
				", line = ", e.line(), ", what = ", e.what());
		} catch(std::exception &e){
			LOG_ERROR("std::exception thrown in job dispatcher: what = ", e.what());
		} catch(...){
			LOG_ERROR("Unknown exception thrown in job dispatcher.");
		}
	}
}
void JobDispatcher::quitModal(){
	atomicStore(g_running, false);
	g_newJobAvail.notify_one();
}

void JobDispatcher::pend(boost::shared_ptr<const JobBase> job){
	{
		const boost::mutex::scoped_lock lock(g_mutex);
		if(g_pool.empty()){
			g_pool.push_front(NULLPTR);
		}
		g_pool.front().swap(job);
		g_queue.splice(g_queue.end(), g_pool, g_pool.begin());
	}
	g_newJobAvail.notify_one();
}
