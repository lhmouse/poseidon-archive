// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2016, LH_Mouse. All wrongs reserved.

#include "precompiled.hpp"
#include "job_promise.hpp"
#include "exception.hpp"
#include "log.hpp"
#include "atomic.hpp"
#include "singletons/job_dispatcher.hpp"

namespace Poseidon {

enum {
	S_LOCKED            = -1,
	S_UNSATISFIED       = 0,
	S_SATISFIED         = 1,
};

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
#ifdef POSEIDON_CXX11
		std::rethrow_exception(m_except);
#else
		boost::rethrow_exception(m_except);
#endif
	}
}

void JobPromise::set_success(){
	const int state = lock();
	if(state != S_UNSATISFIED){
		unlock(state);
		DEBUG_THROW(Exception, sslit("JobPromise is already satisfied"));
	}
	m_except = VAL_INIT;
	unlock(S_SATISFIED);
}
#ifdef POSEIDON_CXX11
void JobPromise::set_exception(std::exception_ptr except)
#else
void JobPromise::set_exception(boost::exception_ptr except)
#endif
{
	assert(except);

	const int state = lock();
	if(state != S_UNSATISFIED){
		unlock(state);
		DEBUG_THROW(Exception, sslit("JobPromise is already satisfied"));
	}
	m_except = STD_MOVE_IDN(except);
	unlock(S_SATISFIED);
}

void yield(boost::shared_ptr<const JobPromise> promise, bool insignificant){
	JobDispatcher::yield(STD_MOVE_IDN(promise), insignificant);
}

}
