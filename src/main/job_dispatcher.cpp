#include "../precompiled.hpp"
#include "job_dispatcher.hpp"
#include "atomic.hpp"
#include "exception.hpp"
#include "log.hpp"
#include <boost/thread.hpp>
#include <deque>
using namespace Poseidon;

namespace {

volatile bool g_running = false;
boost::mutex g_queueMutex;
std::deque<boost::shared_ptr<const JobBase> > g_jobQueue;
boost::condition_variable g_newJobAvail;

}

JobBase::~JobBase(){
}

void JobBase::pend() const {
	const boost::mutex::scoped_lock lock(g_queueMutex);
	g_jobQueue.push_back(shared_from_this());
	g_newJobAvail.notify_one();
}

void JobDispatcher::doModal(){
	if(atomicExchange(g_running, true) != false){
		LOG_FATAL <<"Only one modal loop is allowed at the same time.";
		std::abort();
	}
	for(;;){
		boost::mutex::scoped_lock lock(g_queueMutex);
		while(g_jobQueue.empty()){
			g_newJobAvail.wait(lock);
			if(atomicLoad(g_running) == false){
				return;
			}
		}
		AUTO(const job, g_jobQueue.front());
		g_jobQueue.pop_front();
		lock.unlock();

		try {
			job->perform();
		} catch(Exception &e){
			LOG_ERROR <<"Exception thrown in job dispatcher: file = " <<e.file() <<", line = " <<e.line()
				<<", what = " <<e.what();
		} catch(std::exception &e){
			LOG_ERROR <<"std::exception thrown in job dispatcher: what = " <<e.what();
		} catch(...){
			LOG_ERROR <<"unknown exception thrown in job dispatcher";
		}
	}
}
void JobDispatcher::quitModal(){
	atomicStore(g_running, false);
	g_newJobAvail.notify_all();
}
