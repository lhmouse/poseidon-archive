// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2015, LH_Mouse. All wrongs reserved.

#include "precompiled.hpp"
#include "job_base.hpp"
#include "singletons/job_dispatcher.hpp"

namespace Poseidon {

JobBase::~JobBase(){
}

void enqueue_job(boost::shared_ptr<JobBase> job,
	boost::shared_ptr<const JobPromise> promise, boost::shared_ptr<const bool> withdrawn)
{
	JobDispatcher::enqueue(STD_MOVE(job), STD_MOVE(promise), STD_MOVE(withdrawn));
}
void yield_job(boost::shared_ptr<const JobPromise> promise){
	JobDispatcher::yield(STD_MOVE(promise));
}
void detach_yieldable_job() NOEXCEPT {
	JobDispatcher::detach_yieldable();
}

}
