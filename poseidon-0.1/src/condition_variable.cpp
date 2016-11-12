// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2016, LH_Mouse. All wrongs reserved.

#include "precompiled.hpp"
#include "condition_variable.hpp"
#include <boost/static_assert.hpp>
#include <pthread.h>
#include <time.h>
#include "log.hpp"
#include "system_exception.hpp"

namespace Poseidon {

ConditionVariable::ConditionVariable(){
	::pthread_condattr_t attr;
	int err = ::pthread_condattr_init(&attr);
	if(err != 0){
		LOG_POSEIDON_ERROR("::pthread_condattr_init() failed with error code ", err);
		DEBUG_THROW(SystemException, err);
	}
	err = ::pthread_condattr_setclock(&attr, CLOCK_MONOTONIC);
	if(err != 0){
		::pthread_condattr_destroy(&attr);
		LOG_POSEIDON_ERROR("::pthread_condattr_settype() failed with error code ", err);
		DEBUG_THROW(SystemException, err);
	}
	err = ::pthread_cond_init(&m_cond, &attr);
	if(err != 0){
		::pthread_condattr_destroy(&attr);
		LOG_POSEIDON_ERROR("::pthread_cond_init() failed with error code ", err);
		DEBUG_THROW(SystemException, err);
	}
	::pthread_condattr_destroy(&attr);
}
ConditionVariable::~ConditionVariable(){
	int err = ::pthread_cond_destroy(&m_cond);
	if(err != 0){
		LOG_POSEIDON_ERROR("::pthread_cond_destroy() failed with error code ", err);
	}
}

void ConditionVariable::wait(Mutex::UniqueLock &lock){
	assert(lock.m_locked);

	const int err = ::pthread_cond_wait(&m_cond, &(lock.m_owner->m_mutex));
	if(err != 0){
		LOG_POSEIDON_ERROR("::pthread_cond_wait() failed with error code ", err);
		DEBUG_THROW(SystemException, err);
	}
}
bool ConditionVariable::timed_wait(Mutex::UniqueLock &lock, unsigned long long ms){
	assert(lock.m_locked);

	::timespec tp;
	if(::clock_gettime(CLOCK_MONOTONIC, &tp) != 0){
		const int err = errno;
		LOG_POSEIDON_ERROR("::clock_gettime() failed with error code ", err);
		DEBUG_THROW(SystemException, err);
	}
	tp.tv_sec += static_cast<std::time_t>(ms / 1000);
	tp.tv_nsec += static_cast<long>(ms % 1000 * 1000000);
	if(tp.tv_nsec >= 1000000000){
		++tp.tv_sec;
		tp.tv_nsec -= 1000000000;
	}
	const int err = ::pthread_cond_timedwait(&m_cond, &(lock.m_owner->m_mutex), &tp);
	if(err != 0){
		if(err == ETIMEDOUT){
			return false;
		}
		LOG_POSEIDON_ERROR("::pthread_cond_timedwait() failed with error code ", err);
		DEBUG_THROW(SystemException, err);
	}
	return true;
}

void ConditionVariable::signal() NOEXCEPT {
	const int err = ::pthread_cond_signal(&m_cond);
	if(err != 0){
		LOG_POSEIDON_FATAL("::pthread_cond_signal() failed with error code ", err);
		std::abort();
	}
}
void ConditionVariable::broadcast() NOEXCEPT {
	const int err = ::pthread_cond_broadcast(&m_cond);
	if(err != 0){
		LOG_POSEIDON_FATAL("::pthread_cond_broadcast() failed with error code ", err);
		std::abort();
	}
}

}
