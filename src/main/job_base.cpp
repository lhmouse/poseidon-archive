// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2015, LH_Mouse. All wrongs reserved.

#include "precompiled.hpp"
#include "job_base.hpp"
#include "singletons/job_dispatcher.hpp"

namespace Poseidon {

JobBase::~JobBase(){
}

void pendJob(boost::shared_ptr<JobBase> job){
	JobDispatcher::pend(STD_MOVE(job));
}

}
