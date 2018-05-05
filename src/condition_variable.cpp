// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2018, LH_Mouse. All wrongs reserved.

#include "precompiled.hpp"
#include "condition_variable.hpp"
#include "log.hpp"
#include "errno.hpp"
#include "system_exception.hpp"
#include <boost/static_assert.hpp>
#include <time.h>

namespace Poseidon {

#define TERMINATE_UNLESS(c_, ...)   do { if(c_){ break; } LOG_POSEIDON_FATAL(__VA_ARGS__); std::terminate(); } while(false)

namespace {
	class Condition_variable_attribute : NONCOPYABLE {
	private:
		::pthread_condattr_t m_attr;

	public:
		Condition_variable_attribute(){
			int err = ::pthread_condattr_init(&m_attr);
			DEBUG_THROW_UNLESS(err == 0, System_exception);
			err = ::pthread_condattr_setclock(&m_attr, CLOCK_MONOTONIC);
			TERMINATE_UNLESS(err == 0, "::pthread_condattr_settype() failed with ", err, " (", get_error_desc(err), ")");
		}
		~Condition_variable_attribute(){
			int err = ::pthread_condattr_destroy(&m_attr);
			TERMINATE_UNLESS(err == 0, "::pthread_condattr_destroy() failed with ", err, " (", get_error_desc(err), ")");
		}

	public:
		operator ::pthread_condattr_t *(){
			return &m_attr;
		}
	};
}

Condition_variable::Condition_variable(){
	int err = ::pthread_cond_init(&m_cond, Condition_variable_attribute());
	DEBUG_THROW_UNLESS(err == 0, System_exception, err);
}
Condition_variable::~Condition_variable(){
	int err = ::pthread_cond_destroy(&m_cond);
	TERMINATE_UNLESS(err == 0, "::pthread_cond_destroy() failed with ", err, " (", get_error_desc(err), ")");
}

void Condition_variable::wait(Mutex::Unique_lock &lock){
	TERMINATE_UNLESS(lock.m_target, "No mutex has been assigned to that Unique_lock.");
	TERMINATE_UNLESS(lock.m_locked, "The mutex has not been locked by that Unique_lock.");

	int err = ::pthread_cond_wait(&m_cond, &(lock.m_target->m_mutex));
	DEBUG_THROW_UNLESS(err == 0, System_exception, err);
}
bool Condition_variable::timed_wait(Mutex::Unique_lock &lock, unsigned long long ms){
	TERMINATE_UNLESS(lock.m_target, "No mutex has been assigned to that Unique_lock.");
	TERMINATE_UNLESS(lock.m_locked, "The mutex has not been locked by that Unique_lock.");

	::timespec tp;
	int err = ::clock_gettime(CLOCK_MONOTONIC, &tp);
	DEBUG_THROW_UNLESS(err == 0, System_exception);
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
	DEBUG_THROW_UNLESS(err == 0, System_exception, err);
	return true;
}

void Condition_variable::signal() NOEXCEPT {
	int err = ::pthread_cond_signal(&m_cond);
	TERMINATE_UNLESS(err == 0, "::pthread_cond_signal() failed with ", err, " (", get_error_desc(err), ")");
}
void Condition_variable::broadcast() NOEXCEPT {
	int err = ::pthread_cond_broadcast(&m_cond);
	TERMINATE_UNLESS(err == 0, "::pthread_cond_broadcast() failed with ", err, " (", get_error_desc(err), ")");
}

}
