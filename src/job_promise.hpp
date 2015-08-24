// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2015, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_JOB_PROMISE_HPP_
#define POSEIDON_JOB_PROMISE_HPP_

#include <boost/shared_ptr.hpp>
#include <boost/exception_ptr.hpp>

namespace Poseidon {

class JobPromise : NONCOPYABLE {
private:
	// mutable Mutex m_mutex;
	// bool m_satisfied;
	mutable volatile int m_state;
	boost::exception_ptr m_except;

public:
	JobPromise() NOEXCEPT;
	~JobPromise();

private:
	bool check(int cmp) const NOEXCEPT;
	int lock() const NOEXCEPT;
	void unlock(int state) const NOEXCEPT;

public:
	bool isSatisfied() const NOEXCEPT;
	void checkAndRethrow() const;

	void setSuccess() NOEXCEPT;
	void setException(const boost::exception_ptr &except) NOEXCEPT;
};

}

#endif
