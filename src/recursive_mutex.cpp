// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2017, LH_Mouse. All wrongs reserved.

#include "precompiled.hpp"
#include "recursive_mutex.hpp"
#include <pthread.h>
#include "log.hpp"
#include "system_exception.hpp"

namespace Poseidon {

RecursiveMutex::UniqueLock::UniqueLock()
	: m_target(NULLPTR), m_locked(false)
{
}
RecursiveMutex::UniqueLock::UniqueLock(RecursiveMutex &target, bool locks_target)
	: m_target(&target), m_locked(false)
{
	if(locks_target){
		lock();
	}
}
RecursiveMutex::UniqueLock::~UniqueLock(){
	if(m_locked){
		unlock();
	}
}

bool RecursiveMutex::UniqueLock::is_locked() const NOEXCEPT {
	return m_locked;
}
void RecursiveMutex::UniqueLock::lock() NOEXCEPT {
	assert(m_target);
	assert(!m_locked);

	const int err = ::pthread_mutex_lock(&(m_target->m_mutex));
	if(err != 0){
		LOG_POSEIDON_FATAL("::pthread_mutex_lock() failed with error code ", err);
		std::abort();
	}
	m_locked = true;
}
void RecursiveMutex::UniqueLock::unlock() NOEXCEPT {
	assert(m_target);
	assert(m_locked);

	const int err = ::pthread_mutex_unlock(&(m_target->m_mutex));
	if(err != 0){
		LOG_POSEIDON_FATAL("::pthread_mutex_unlock() failed with error code ", err);
		std::abort();
	}
	m_locked = false;
}

RecursiveMutex::RecursiveMutex(){
	::pthread_mutexattr_t attr;
	int err = ::pthread_mutexattr_init(&attr);
	if(err != 0){
		LOG_POSEIDON_ERROR("::pthread_mutexattr_init() failed with error code ", err);
		DEBUG_THROW(SystemException, err);
	}
	err = ::pthread_mutexattr_settype(&attr, PTHREAD_MUTEX_RECURSIVE);
	if(err != 0){
		::pthread_mutexattr_destroy(&attr);
		LOG_POSEIDON_ERROR("::pthread_mutexattr_settype() failed with error code ", err);
		DEBUG_THROW(SystemException, err);
	}
	err = ::pthread_mutex_init(&m_mutex, &attr);
	if(err != 0){
		::pthread_mutexattr_destroy(&attr);
		LOG_POSEIDON_ERROR("::pthread_mutex_init() failed with error code ", err);
		DEBUG_THROW(SystemException, err);
	}
	::pthread_mutexattr_destroy(&attr);
}
RecursiveMutex::~RecursiveMutex(){
	int err = ::pthread_mutex_destroy(&m_mutex);
	if(err != 0){
		LOG_POSEIDON_ERROR("::pthread_mutex_destroy() failed with error code ", err);
	}
}

}
