// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2018, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_CONDITION_VARIABLE_HPP_
#define POSEIDON_CONDITION_VARIABLE_HPP_

#include "cxx_util.hpp"
#include "mutex.hpp"
#include <pthread.h>

namespace Poseidon {

class Condition_variable : NONCOPYABLE {
private:
	::pthread_cond_t m_cond;

public:
	Condition_variable();
	~Condition_variable();

public:
	void wait(Mutex::Unique_lock &lock);
	// 返回 true 若条件触发，返回 false 若超时。
	bool timed_wait(Mutex::Unique_lock &lock, unsigned long long ms);

	void signal() NOEXCEPT;
	void broadcast() NOEXCEPT;
};

}

#endif
