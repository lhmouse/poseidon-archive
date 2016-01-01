// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2016, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_JOB_PROMISE_HPP_
#define POSEIDON_JOB_PROMISE_HPP_

#include "cxx_util.hpp"
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
	bool is_satisfied() const NOEXCEPT;
	void check_and_rethrow() const;

	void set_success();
	void set_exception(const boost::exception_ptr &except);
};

}

#endif
