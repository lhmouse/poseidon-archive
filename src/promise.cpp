// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2017, LH_Mouse. All wrongs reserved.

#include "precompiled.hpp"
#include "promise.hpp"
#include "exception.hpp"
#include "log.hpp"
#include "singletons/job_dispatcher.hpp"

namespace Poseidon {

Promise::Promise() NOEXCEPT
	: m_satisfied(false), m_except()
{ }
Promise::~Promise(){
	if(!m_satisfied){
		LOG_POSEIDON_WARNING("Destroying an unsatisfied Promise.");
	}
}

void Promise::check_and_rethrow() const {
	const RecursiveMutex::UniqueLock lock(m_mutex);
	if(!m_satisfied){
		DEBUG_THROW(Exception, sslit("Promise has not been satisfied"));
	}
	if(m_except){
		STD_RETHROW_EXCEPTION(m_except);
	}
}

void Promise::set_success(bool throw_if_already_set){
	const RecursiveMutex::UniqueLock lock(m_mutex);
	if(m_satisfied){
		if(throw_if_already_set){
			DEBUG_THROW(Exception, sslit("Promise has already been satisfied"));
		}
		return;
	}
	m_satisfied = true;
	m_except = NULLPTR;
}
void Promise::set_exception(STD_EXCEPTION_PTR except, bool throw_if_already_set){
	const RecursiveMutex::UniqueLock lock(m_mutex);
	if(m_satisfied){
		if(throw_if_already_set){
			DEBUG_THROW(Exception, sslit("Promise has already been satisfied"));
		}
		return;
	}
	m_satisfied = true;
	m_except = STD_MOVE_IDN(except);
}

void yield(const boost::shared_ptr<const Promise> &promise, bool insignificant){
	JobDispatcher::yield(promise, insignificant);
}

}
