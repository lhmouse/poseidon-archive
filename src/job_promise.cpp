// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2017, LH_Mouse. All wrongs reserved.

#include "precompiled.hpp"
#include "job_promise.hpp"
#include "exception.hpp"
#include "log.hpp"
#include "singletons/job_dispatcher.hpp"

namespace Poseidon {

JobPromise::JobPromise() NOEXCEPT
	: m_satisfied(false), m_except()
{ }
JobPromise::~JobPromise(){
	if(!m_satisfied){
		LOG_POSEIDON_WARNING("Destroying an unsatisfied JobPromise.");
	}
}

bool JobPromise::would_throw() const NOEXCEPT {
	const RecursiveMutex::UniqueLock lock(m_mutex);
	if(!m_satisfied){
		return true;
	}
	if(m_except){
		return true;
	}
	return false;
}
void JobPromise::check_and_rethrow() const {
	const RecursiveMutex::UniqueLock lock(m_mutex);
	if(!m_satisfied){
		DEBUG_THROW(Exception, sslit("JobPromise has not been satisfied"));
	}
	if(m_except){
#ifdef POSEIDON_CXX11
		std::rethrow_exception(m_except);
#else
		boost::rethrow_exception(m_except);
#endif
	}
}

void JobPromise::set_success(){
	const RecursiveMutex::UniqueLock lock(m_mutex);
	if(m_satisfied){
		DEBUG_THROW(Exception, sslit("JobPromise has already been satisfied"));
	}
	m_satisfied = true;
//	m_except = VAL_INIT;
}
#ifdef POSEIDON_CXX11
void JobPromise::set_exception(std::exception_ptr except)
#else
void JobPromise::set_exception(boost::exception_ptr except)
#endif
{
	const RecursiveMutex::UniqueLock lock(m_mutex);
	if(m_satisfied){
		DEBUG_THROW(Exception, sslit("JobPromise has already been satisfied"));
	}
	m_satisfied = true;
	m_except = STD_MOVE_IDN(except);
}

void yield(const boost::shared_ptr<const JobPromise> &promise, bool insignificant){
	JobDispatcher::yield(promise, insignificant);
}

}
