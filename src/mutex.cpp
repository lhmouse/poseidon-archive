// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2017, LH_Mouse. All wrongs reserved.

#include "precompiled.hpp"
#include "mutex.hpp"
#include "log.hpp"
#include "system_exception.hpp"

namespace Poseidon {

#define ABORT_UNLESS(c_, ...)   do { if(c_){ break; } LOG_POSEIDON_FATAL(__VA_ARGS__); } while(false)

namespace {
	class MutexAttribute : NONCOPYABLE {
	private:
		::pthread_mutexattr_t m_attr;

	public:
		MutexAttribute(){
			int err = ::pthread_mutexattr_init(&m_attr);
			DEBUG_THROW_UNLESS(err == 0, SystemException);
			err = ::pthread_mutexattr_settype(&m_attr, PTHREAD_MUTEX_ERRORCHECK);
			ABORT_UNLESS(err == 0, "::pthread_mutexattr_settype() failed with error code ", err);
		}
		~MutexAttribute(){
			int err = ::pthread_mutexattr_destroy(&m_attr);
			ABORT_UNLESS(err == 0, "::pthread_mutexattr_destroy() failed with error code ", err);
		}

	public:
		operator ::pthread_mutexattr_t *(){
			return &m_attr;
		}
	};
}

Mutex::UniqueLock::UniqueLock()
	: m_target(NULLPTR), m_locked(false)
{ }
Mutex::UniqueLock::UniqueLock(Mutex &target, bool locks_target)
	: m_target(&target), m_locked(false)
{
	if(locks_target){
		lock();
	}
}
Mutex::UniqueLock::~UniqueLock(){
	if(m_locked){
		unlock();
	}
}

bool Mutex::UniqueLock::is_locked() const NOEXCEPT {
	return m_locked;
}
void Mutex::UniqueLock::lock() NOEXCEPT {
	ABORT_UNLESS(m_target, "No mutex has been assigned to this UniqueLock.");
	ABORT_UNLESS(!m_locked, "The mutex has already been locked by this UniqueLock.");

	int err = ::pthread_mutex_lock(&(m_target->m_mutex));
	ABORT_UNLESS(err == 0, "::pthread_mutex_lock() failed with error code ", err);
	m_locked = true;
}
void Mutex::UniqueLock::unlock() NOEXCEPT {
	ABORT_UNLESS(m_target, "No mutex has been assigned to this UniqueLock.");
	ABORT_UNLESS(m_locked, "The mutex has not already been locked by this UniqueLock.");

	int err = ::pthread_mutex_unlock(&(m_target->m_mutex));
	ABORT_UNLESS(err == 0, "::pthread_mutex_unlock() failed with error code ", err);
	m_locked = false;
}

Mutex::Mutex(){
	int err = ::pthread_mutex_init(&m_mutex, MutexAttribute());
	DEBUG_THROW_UNLESS(err == 0, SystemException, err);
}
Mutex::~Mutex(){
	int err = ::pthread_mutex_destroy(&m_mutex);
	ABORT_UNLESS(err == 0, "::pthread_mutex_destroy() failed with error code ", err);
}

}
