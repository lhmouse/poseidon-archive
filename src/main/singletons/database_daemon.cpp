#include "../../precompiled.hpp"
#include "database_daemon.hpp"
#include "../database/object.hpp"
#include <queue>
#include <boost/thread.hpp>
#include <boost/make_shared.hpp>
#include "../exception.hpp"
#include "../log.hpp"
#include "../atomic.hpp"
using namespace Poseidon;

namespace {

volatile bool g_daemonRunning = false;
boost::thread g_daemonThread;
boost::mutex g_queueMutex;
std::queue<boost::shared_ptr<const DatabaseObject> > g_dirtyQueue;
boost::condition_variable g_dirtyAvail;

void threadProc(){
	LOG_INFO("Database daemon started.");

	for(;;){
		try {
			boost::shared_ptr<const DatabaseObject> dbObj;
			{
				boost::mutex::scoped_lock lock(g_queueMutex);
				for(;;){
					if(!g_dirtyQueue.empty()){
						dbObj = STD_MOVE(g_dirtyQueue.front());
						g_dirtyQueue.pop();
						break;
					}
					if(!atomicLoad(g_daemonRunning)){
						break;
					}
					g_dirtyAvail.wait(lock);
				}
			}
			if(!dbObj){
				break;
			}
			//job->perform();
		} catch(Exception &e){
			LOG_ERROR("Exception thrown in database daemon: file = ", e.file(),
				", line = ", e.line(), ": what = ", e.what());
		} catch(std::exception &e){
			LOG_ERROR("std::exception thrown in database daemon: what = ", e.what());
		} catch(...){
			LOG_ERROR("Unknown exception thrown in database daemon.");
		}
	}

	LOG_INFO("Database daemon stopped.");
}

}

void DatabaseDaemon::start(){
	if(atomicExchange(g_daemonRunning, true) != false){
		LOG_FATAL("Only one daemon is allowed at the same time.");
		std::abort();
	}
	LOG_INFO("Starting database daemon...");

	boost::thread(threadProc).swap(g_daemonThread);
}
void DatabaseDaemon::stop(){
	LOG_INFO("Stopping database daemon...");

	atomicStore(g_daemonRunning, false);
	g_dirtyAvail.notify_one();
	if(g_daemonThread.joinable()){
		g_daemonThread.join();
	}
}
