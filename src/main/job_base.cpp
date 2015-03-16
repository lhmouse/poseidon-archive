// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2015, LH_Mouse. All wrongs reserved.

#include "precompiled.hpp"
#include "job_base.hpp"
#include "singletons/job_dispatcher.hpp"
#include "exception.hpp"

namespace Poseidon {

JobBase::TryAgainLater::TryAgainLater(boost::shared_ptr<const void> context) NOEXCEPT
	: m_context(STD_MOVE(context))
{
}
JobBase::TryAgainLater::~TryAgainLater() NOEXCEPT {
}

JobBase::~JobBase(){
}

void enqueueJob(boost::shared_ptr<const JobBase> job, boost::uint64_t delay){
	JobDispatcher::enqueue(STD_MOVE(job), delay);
}
void suspendCurrentJob(boost::shared_ptr<const void> context){
	throw JobBase::TryAgainLater(STD_MOVE(context));
}

}
