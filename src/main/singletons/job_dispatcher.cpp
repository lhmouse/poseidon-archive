#include "../../precompiled.hpp"
#include "job_dispatcher.hpp"
#include "../atomic.hpp"
#include "../exception.hpp"
#include "../log.hpp"
#include <boost/thread.hpp>
#include <list>
using namespace Poseidon;

namespace {

volatile bool g_running = false;
boost::mutex g_queueMutex;
std::list<boost::shared_ptr<const JobBase> > g_queue, g_pool;
boost::condition_variable g_newJobAvail;

}

JobBase::~JobBase(){
}

void JobBase::pend() const {
	{
		const boost::mutex::scoped_lock lock(g_queueMutex);
		if(g_pool.empty()){
			g_queue.push_back(boost::shared_ptr<const JobBase>());
		} else {
			g_queue.splice(g_queue.end(), g_pool, g_pool.begin());
		}
		g_queue.back() = virtualSharedFromThis<JobBase>();
	}
	g_newJobAvail.notify_one();
}

void JobDispatcher::doModal(){
	if(atomicExchange(g_running, true) != false){
		LOG_FATAL, "Only one modal loop is allowed at the same time.";
		std::abort();
	}
	for(;;){
		boost::shared_ptr<const JobBase> job;
		{
			boost::mutex::scoped_lock lock(g_queueMutex);
			while(g_queue.empty()){
				g_newJobAvail.wait(lock);
				if(atomicLoad(g_running) == false){
					return;
				}
			}
			job = g_queue.front();
			g_queue.front().reset();
			g_pool.splice(g_pool.begin(), g_queue, g_queue.begin());
		}

		try {
			job->perform();
		} catch(Exception &e){
			LOG_ERROR, "Exception thrown in job dispatcher: file = ", e.file(), ", line = ", e.line(), ": what = ", e.what();
		} catch(std::exception &e){
			LOG_ERROR, "std::exception thrown in job dispatcher: what = ", e.what();
		} catch(...){
			LOG_ERROR, "Unknown exception thrown in job dispatcher";
		}
	}
}
void JobDispatcher::quitModal(){
	atomicStore(g_running, false);
	g_newJobAvail.notify_all();
}
