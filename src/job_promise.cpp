// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2015, LH_Mouse. All wrongs reserved.

#include "precompiled.hpp"
#include "job_promise.hpp"
#include "exception.hpp"
#include "log.hpp"
#include "atomic.hpp"

namespace Poseidon {

namespace {
	enum {
		S_LOCKED			= -1,
		S_UNSATISFIED		= 0,
		S_SATISFIED			= 1,
	};
}

JobPromise::JobPromise() NOEXCEPT
	: m_state(S_UNSATISFIED)
{
}
JobPromise::~JobPromise(){
}

bool JobPromise::check(int cmp) const NOEXCEPT {
	for(;;){
		const int state = atomic_load(m_state, ATOMIC_CONSUME);
		if(state != S_LOCKED){
			return state == cmp;
		}
		atomic_pause();
	}
}
int JobPromise::lock() const NOEXCEPT {
	for(;;){
		const int state = atomic_exchange(m_state, S_LOCKED, ATOMIC_SEQ_CST);
		if(state != S_LOCKED){
			return state;
		}
		atomic_pause();
	}
}
void JobPromise::unlock(int state) const NOEXCEPT {
	atomic_store(m_state, state, ATOMIC_SEQ_CST);
}

bool JobPromise::is_satisfied() const NOEXCEPT {
	return !check(S_UNSATISFIED);
}
void JobPromise::check_and_rethrow() const {
	const int state = lock();
	if(state == S_UNSATISFIED){
		unlock(state);
		DEBUG_THROW(Exception, sslit("JobPromise is not satisfied"));
	}
	const AUTO(except, m_except);
	unlock(state);

	if(except){
		boost::rethrow_exception(m_except);
	}
}

void JobPromise::set_success(){
	const int state = lock();
	if(state != S_UNSATISFIED){
		unlock(state);
		DEBUG_THROW(Exception, sslit("JobPromise is already satisfied"));
	}
	m_except = boost::exception_ptr();
	unlock(S_SATISFIED);
}
void JobPromise::set_exception(const boost::exception_ptr &except){
	assert(except);

	const int state = lock();
	if(state != S_UNSATISFIED){
		unlock(state);
		DEBUG_THROW(Exception, sslit("JobPromise is already satisfied"));
	}
	m_except = except;
	unlock(S_SATISFIED);
}

}
