// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2015, LH_Mouse. All wrongs reserved.

#include "precompiled.hpp"
#include "job_base.hpp"
#include "singletons/job_dispatcher.hpp"
#include "exception.hpp"

namespace Poseidon {

JobBase::TryAgainLater::~TryAgainLater() NOEXCEPT {
}

const char *JobBase::TryAgainLater::what() const NOEXCEPT {
	return "Poseidon::TryAgainLater";
}

JobBase::~JobBase(){
}

void enqueueJob(boost::shared_ptr<JobBase> job){
	JobDispatcher::enqueue(STD_MOVE(job));
}
void suspendCurrentJob(){
	throw JobBase::TryAgainLater();
}

}
