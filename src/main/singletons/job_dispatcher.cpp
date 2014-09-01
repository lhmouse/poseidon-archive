#include "../../precompiled.hpp"
#include "job_dispatcher.hpp"
#include "../atomic.hpp"
#include "../exception.hpp"
#include "../log.hpp"
#include <boost/thread/mutex.hpp>
#include <boost/thread/condition_variable.hpp>
#include <queue>
using namespace Poseidon;

namespace {

volatile bool g_running = false;
boost::mutex g_queueMutex;
std::queue<boost::shared_ptr<const JobBase> > g_queue;
boost::condition_variable g_newJobAvail;

}

JobBase::~JobBase(){
}

void JobBase::pend() const {
	{
		const boost::mutex::scoped_lock lock(g_queueMutex);
		g_queue.push(virtualSharedFromThis<JobBase>());
	}
	g_newJobAvail.notify_one();
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
				boost::mutex::scoped_lock lock(g_queueMutex);
				for(;;){
					if(!atomicLoad(g_running)){
						break;
					}
					if(!g_queue.empty()){
						job = STD_MOVE(g_queue.front());
						g_queue.pop();
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
				", line = ", e.line(), ": what = ", e.what());
		} catch(std::exception &e){
			LOG_ERROR("std::exception thrown in job dispatcher: what = ", e.what());
		} catch(...){
			LOG_ERROR("Unknown exception thrown in job dispatcher");
		}
	}
}
void JobDispatcher::quitModal(){
	atomicStore(g_running, false);
	g_newJobAvail.notify_one();
}
