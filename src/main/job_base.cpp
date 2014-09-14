#include "../precompiled.hpp"
#include "job_base.hpp"
#include "singletons/job_dispatcher.hpp"
using namespace Poseidon;

JobBase::~JobBase(){
}

void JobBase::pend() const {
	JobDispatcher::pend(virtualShared<JobBase>());
}
