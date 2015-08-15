// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2015, LH_Mouse. All wrongs reserved.

#include "precompiled.hpp"
#include "job_base.hpp"
#include "singletons/job_dispatcher.hpp"

namespace Poseidon {

JobBase::~JobBase(){
}

void enqueueJob(boost::shared_ptr<const JobBase> job,
	boost::function<bool ()> pred, boost::shared_ptr<const bool> withdrawn)
{
	JobDispatcher::enqueue(STD_MOVE(job), STD_MOVE(pred), STD_MOVE(withdrawn));
}
void yieldJob(boost::function<bool ()> pred){
	JobDispatcher::yield(STD_MOVE(pred));
}
void detachYieldableJob() NOEXCEPT {
	JobDispatcher::detachYieldable();
}

}
