// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2018, LH_Mouse. All wrongs reserved.

#include "precompiled.hpp"
#include "job_base.hpp"
#include "singletons/job_dispatcher.hpp"

namespace Poseidon {

JobBase::~JobBase(){ }

void enqueue(boost::shared_ptr<JobBase> job, boost::shared_ptr<const bool> withdrawn){
	JobDispatcher::enqueue(STD_MOVE(job), STD_MOVE(withdrawn));
}

}
