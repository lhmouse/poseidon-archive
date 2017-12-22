// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2017, LH_Mouse. All wrongs reserved.

#include "precompiled.hpp"
#include "condition_variable.hpp"
#include <boost/static_assert.hpp>
#include <time.h>
#include "log.hpp"
#include "system_exception.hpp"

namespace Poseidon {

#define ABORT_UNLESS(c_, ...)   do { if(c_){ break; } LOG_POSEIDON_FATAL(__VA_ARGS__); std::abort(); } while(false)

namespace {
	class ConditionVariableAttribute : NONCOPYABLE {
	private:
		::pthread_condattr_t m_attr;

	public:
		ConditionVariableAttribute(){
			int err = ::pthread_condattr_init(&m_attr);
			DEBUG_THROW_UNLESS(err == 0, SystemException);
			err = ::pthread_condattr_setclock(&m_attr, CLOCK_MONOTONIC);
			ABORT_UNLESS(err == 0, "::pthread_condattr_settype() failed with error code ", err);
		}
		~ConditionVariableAttribute(){
			int err = ::pthread_condattr_destroy(&m_attr);
			ABORT_UNLESS(err == 0, "::pthread_condattr_destroy() failed with error code ", err);
		}

	public:
		operator ::pthread_condattr_t *(){
			return &m_attr;
		}
	};
}

ConditionVariable::ConditionVariable(){
	int err = ::pthread_cond_init(&m_cond, ConditionVariableAttribute());
	DEBUG_THROW_UNLESS(err == 0, SystemException, err);
}
ConditionVariable::~ConditionVariable(){
	int err = ::pthread_cond_destroy(&m_cond);
	ABORT_UNLESS(err == 0, "::pthread_cond_destroy() failed with error code ", err);
}

void ConditionVariable::wait(Mutex::UniqueLock &lock){
	ABORT_UNLESS(lock.m_target, "No mutex has been assigned to that UniqueLock.");
	ABORT_UNLESS(lock.m_locked, "The mutex has not been locked by that UniqueLock.");

	int err = ::pthread_cond_wait(&m_cond, &(lock.m_target->m_mutex));
	DEBUG_THROW_UNLESS(err == 0, SystemException, err);
}
bool ConditionVariable::timed_wait(Mutex::UniqueLock &lock, unsigned long long ms){
	ABORT_UNLESS(lock.m_target, "No mutex has been assigned to that UniqueLock.");
	ABORT_UNLESS(lock.m_locked, "The mutex has not been locked by that UniqueLock.");

	::timespec tp;
	int err = ::clock_gettime(CLOCK_MONOTONIC, &tp);
	DEBUG_THROW_UNLESS(err == 0, SystemException);
	tp.tv_sec += static_cast<std::time_t>(ms / 1000);
	tp.tv_nsec += static_cast<long>(ms % 1000 * 1000000);
	if(tp.tv_nsec >= 1000000000){
		++tp.tv_sec;
		tp.tv_nsec -= 1000000000;
	}
	err = ::pthread_cond_timedwait(&m_cond, &(lock.m_target->m_mutex), &tp);
	if(err == ETIMEDOUT){
		return false;
	}
	DEBUG_THROW_UNLESS(err == 0, SystemException, err);
	return true;
}

void ConditionVariable::signal() NOEXCEPT {
	int err = ::pthread_cond_signal(&m_cond);
	ABORT_UNLESS(err == 0, "::pthread_cond_signal() failed with error code ", err);
}
void ConditionVariable::broadcast() NOEXCEPT {
	int err = ::pthread_cond_broadcast(&m_cond);
	ABORT_UNLESS(err == 0, "::pthread_cond_broadcast() failed with error code ", err);
}

}
